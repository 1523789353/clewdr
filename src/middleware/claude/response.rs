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
    types::claude::{CreateMessageResponse, StreamEvent},
};
use futures::StreamExt;
use crate::types::claude::{ContentBlock, CreateMessageResponseDelta, StopReason, Usage};


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
async fn aggregate_stream(resp: Response) -> Result<CreateMessageResponse, Response> {
    let mut completion = String::new();
    let mut response = CreateMessageResponse::default();
    let mut usage = Usage::default();
    let mut stop_reason = None;

    let stream = resp.into_body().into_data_stream().eventsource();
    let mut stream = Box::pin(stream);

    while let Some(Ok(event)) = stream.next().await {
        if event.event == "error" {
            warn!("[PSEUDO] SSE error: {}", event.data);
            continue;
        }
        tracing::info!("[PSEUDO] Upstream chunk received: {}", event.data);
        let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) else {
            continue;
        };
        match parsed {
            StreamEvent::MessageStart { message } => {
                response = message;
            }
            StreamEvent::ContentBlockDelta { delta, .. } => {
                if let crate::types::claude::ContentBlockDelta::TextDelta { text } = delta {
                    completion.push_str(&text);
                }
            }
            StreamEvent::MessageDelta { delta, usage: new_usage } => {
                if let Some(reason) = delta.stop_reason {
                    stop_reason = Some(reason);
                }
                if let Some(u) = new_usage {
                    usage = u;
                }
            }
            StreamEvent::MessageStop { .. } => {
                // The final usage is often in MessageDelta, but we can confirm here
            }
            _ => (),
        }
    }

    if let Some(last_block) = response.content.last_mut() {
        if let ContentBlock::Text { text } = last_block {
            *text = completion;
        } else {
            response.content.push(ContentBlock::Text { text: completion });
        }
    } else {
        response.content.push(ContentBlock::Text { text: completion });
    }

    response.stop_reason = stop_reason;
    response.usage = Some(usage);
    // trick to get the correct output tokens
    let output_tokens = response.count_tokens();
    if let Some(u) = response.usage.as_mut() {
        u.output_tokens = output_tokens;
    }

    tracing::info!("[PSEUDO] Downstream non-stream response created.");
    crate::utils::print_out_json(&response, "pseudo_downstream_resp.json");

    Ok(response)
}
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
