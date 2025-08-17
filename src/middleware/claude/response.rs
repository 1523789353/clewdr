use axum::{
    Json,
    body::{self, Body},
    response::{IntoResponse, Response, Sse},
};
use eventsource_stream::Eventsource;
use futures::TryStreamExt;
use http::header::CONTENT_TYPE;
use tracing::warn;

use super::{ClaudeApiFormat, transform_stream};
use crate::{
    middleware::claude::{ClaudeContext, transforms_json},
    types::claude::{ContentBlock, CreateMessageResponse, StreamEvent, Usage},
};
use futures::StreamExt;

async fn aggregate_stream(resp: Response) -> Result<CreateMessageResponse, Response> {
    let mut response = CreateMessageResponse::default();
    let mut usage = Usage::default();
    let mut stop_reason = None;
    let mut content_blocks: Vec<ContentBlock> = Vec::new();

    let stream = resp.into_body().into_data_stream().eventsource();
    let mut stream = Box::pin(stream);

    while let Some(Ok(event)) = stream.next().await {
        if event.event == "error" {
            warn!("SSE error: {}", event.data);
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) else {
            continue;
        };
        match parsed {
            StreamEvent::MessageStart { message } => {
                response.id = message.id;
                response.type_ = message.type_;
                response.role = message.role;
                response.model = message.model;
                if let Some(u) = message.usage {
                    usage.input_tokens = u.input_tokens;
                }
            }
            StreamEvent::ContentBlockStart { index, content_block } => {
                if index >= content_blocks.len() {
                    content_blocks.resize(index + 1, ContentBlock::Text { text: String::new() }); // Placeholder
                }
                content_blocks[index] = content_block;
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = content_blocks.get_mut(index) {
                    match (block, delta) {
                        (ContentBlock::Text { text }, crate::types::claude::ContentBlockDelta::TextDelta { text: delta_text }) => {
                            text.push_str(&delta_text);
                        }
                        (ContentBlock::Thinking { thinking, .. }, crate::types::claude::ContentBlockDelta::ThinkingDelta { thinking: delta_thinking }) => {
                            thinking.push_str(&delta_thinking);
                        }
                        _ => {}
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage: new_usage } => {
                if let Some(reason) = delta.stop_reason {
                    stop_reason = Some(reason);
                }
                if let Some(u) = new_usage {
                    usage.output_tokens += u.output_tokens;
                }
            }
            _ => (),
        }
    }

    response.content = content_blocks;
    response.stop_reason = stop_reason;

    usage.output_tokens = response.count_tokens();
    response.usage = Some(usage);

    Ok(response)
}


async fn parse_response<T>(resp: Response) -> Result<T, Response>
where
    T: serde::de::DeserializeOwned,
{
    let body = body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .inspect_err(|err| {
            warn!("Failed to read response body: {}", err);
        })
        .unwrap_or_default();
    let Ok(parsed) = serde_json::from_slice::<T>(&body) else {
        return Err(Response::builder()
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap());
    };
    Ok(parsed)
}

/// Transforms responses to ensure compatibility with the OpenAI API format
///
/// This middleware function analyzes responses and transforms them when necessary
/// to ensure compatibility between Claude and OpenAI API formats, particularly
/// for streaming responses. If the response is:
///
/// - From the Claude API format: No transformation needed
/// - Not streaming: No transformation needed
/// - Has a non-200 status code: No transformation needed
/// - OpenAI format and streaming: Transforms the stream to match OpenAI event format
///
/// # Arguments
///
/// * `resp` - The original response to be potentially transformed
///
/// # Returns
///
/// The original or transformed response as appropriate
pub async fn to_oai(resp: Response) -> impl IntoResponse {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };
    if ClaudeApiFormat::Claude == cx.api_format() {
        return resp;
    }

    if cx.pseudo_non_stream() {
        return match aggregate_stream(resp).await {
            Ok(response) => Json(transforms_json(response)).into_response(),
            Err(resp) => resp,
        };
    }
    if !cx.is_stream() {
        match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => return Json(transforms_json(response)).into_response(),
            Err(resp) => return resp,
        }
    }
    let stream = resp.into_body().into_data_stream().eventsource();
    let stream = transform_stream(stream);
    Sse::new(stream)
        .keep_alive(Default::default())
        .into_response()
}

pub async fn add_usage_info(resp: Response) -> impl IntoResponse {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };

    if cx.pseudo_non_stream() {
        return match aggregate_stream(resp).await {
            Ok(response) => Json(response).into_response(),
            Err(resp) => resp,
        };
    }

    let (mut usage, stream) = (cx.usage().to_owned(), cx.is_stream());
    if !stream {
        let mut response = match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => response,
            Err(resp) => return resp,
        };
        let output_tokens = response.count_tokens();
        usage.output_tokens = output_tokens;
        response.usage = Some(usage);
        return Json(response).into_response();
    }
    let stream = resp
        .into_body()
        .into_data_stream()
        .eventsource()
        .map_ok(move |event| {
            let new_event = axum::response::sse::Event::default()
                .event(event.event)
                .id(event.id);
            let new_event = if let Some(retry) = event.retry {
                new_event.retry(retry)
            } else {
                new_event
            };
            let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) else {
                return new_event.data(event.data);
            };
            match parsed {
                StreamEvent::MessageStart { mut message } => {
                    message.usage.get_or_insert(usage.to_owned());
                    new_event
                        .json_data(StreamEvent::MessageStart { message })
                        .unwrap()
                }
                StreamEvent::MessageDelta { delta, usage } => {
                    let usage = usage.unwrap_or_default();
                    new_event
                        .json_data(StreamEvent::MessageDelta {
                            delta,
                            usage: Some(usage),
                        })
                        .unwrap()
                }
                _ => new_event.data(event.data),
            }
        });

    Sse::new(stream)
        .keep_alive(Default::default())
        .into_response()
}

pub async fn check_overloaded(mut resp: Response) -> Response {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };
    if !cx.is_stream() {
        return resp;
    }
    if resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| !v.contains("text/event-stream"))
    {
        resp.extensions_mut().remove::<ClaudeContext>();
    }
    resp
}
