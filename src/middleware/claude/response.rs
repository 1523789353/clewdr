use std::sync::{Arc, Mutex};

use axum::{
    body::{self, Body},
    response::{IntoResponse, Response, Sse},
    Json,
};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures::TryStreamExt;
use http::header::CONTENT_TYPE;
use tiktoken_rs::o200k_base;
use tracing::warn;

use super::{ClaudeApiFormat, transform_stream};
use crate::{
    middleware::claude::{transforms_json, ClaudeContext},
    types::claude::{ContentBlock, CreateMessageResponse, StreamEvent},
};

async fn aggregate_stream(
    resp: Response,
    cx: &ClaudeContext,
) -> Result<CreateMessageResponse, Response> {
    let mut response = CreateMessageResponse::default();
    let mut usage = cx.usage().to_owned();
    usage.output_tokens = 0; // Reset output tokens before accumulating

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
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if index >= content_blocks.len() {
                    content_blocks.resize(
                        index + 1,
                        ContentBlock::Text {
                            text: String::new(),
                        },
                    ); // Placeholder
                }
                content_blocks[index] = content_block;
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = content_blocks.get_mut(index) {
                    match (block, delta) {
                        (
                            ContentBlock::Text { text },
                            crate::types::claude::ContentBlockDelta::TextDelta { text: delta_text },
                        ) => {
                            text.push_str(&delta_text);
                        }
                        (
                            ContentBlock::Thinking { thinking, .. },
                            crate::types::claude::ContentBlockDelta::ThinkingDelta {
                                thinking: delta_thinking,
                            },
                        ) => {
                            thinking.push_str(&delta_thinking);
                        }
                        _ => {}
                    }
                }
            }
            StreamEvent::MessageDelta {
                delta,
                usage: new_usage,
            } => {
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

    // Recalculate output tokens as a fallback
    let final_output_tokens = response.count_tokens();
    if usage.output_tokens == 0 && final_output_tokens > 0 {
        usage.output_tokens = final_output_tokens;
    }

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

pub async fn to_oai(resp: Response) -> impl IntoResponse {
    let cx_clone = resp.extensions().get::<ClaudeContext>().cloned();

    let Some(cx) = cx_clone else {
        return resp.into_response();
    };

    if ClaudeApiFormat::Claude == cx.api_format() {
        return resp.into_response();
    }

    if cx.pseudo_non_stream() {
        return match aggregate_stream(resp, &cx).await {
            Ok(response) => Json(transforms_json(response)).into_response(),
            Err(resp) => resp,
        };
    }
    if !cx.is_stream() {
        return match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => Json(transforms_json(response)).into_response(),
            Err(resp) => resp,
        };
    }

    let stream = resp.into_body().into_data_stream().eventsource();
    let stream = transform_stream(stream);
    Sse::new(stream)
        .keep_alive(Default::default())
        .into_response()
}

pub async fn add_usage_info(resp: Response) -> impl IntoResponse {
    let cx_clone = resp.extensions().get::<ClaudeContext>().cloned();

    let Some(cx) = cx_clone else {
        return resp.into_response();
    };

    if cx.pseudo_non_stream() {
        return match aggregate_stream(resp, &cx).await {
            Ok(response) => Json(response).into_response(),
            Err(resp) => resp,
        };
    }

    let (mut usage_from_context, stream) = (cx.usage().to_owned(), cx.is_stream());
    if !stream {
        let mut response = match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => response,
            Err(resp) => return resp,
        };
        let output_tokens = response.count_tokens();
        usage_from_context.output_tokens = output_tokens;
        response.usage = Some(usage_from_context);
        return Json(response).into_response();
    }

    let completion = Arc::new(Mutex::new(String::new()));

    let stream = resp
        .into_body()
        .into_data_stream()
        .eventsource()
        .map_ok(move |event| {
            let completion = completion.clone();
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
                    message.usage = Some(usage_from_context.clone());
                    new_event
                        .json_data(StreamEvent::MessageStart { message })
                        .unwrap()
                }
                StreamEvent::ContentBlockDelta { delta, .. } => {
                    if let crate::types::claude::ContentBlockDelta::TextDelta { text } = delta {
                        completion.lock().unwrap().push_str(&text);
                    }
                    new_event.data(event.data)
                }
                StreamEvent::MessageDelta { delta, mut usage } => {
                    if delta.stop_reason.is_some() {
                        let mut final_usage = usage.unwrap_or_default();
                        if final_usage.output_tokens == 0 {
                            let bpe = o200k_base().expect("Failed to get tiktoken");
                            let calculated_output = bpe.encode_with_special_tokens(&completion.lock().unwrap()).len() as u32;
                            if calculated_output > 0 {
                                final_usage.output_tokens = calculated_output;
                            }
                        }
                        // Always carry over the input_tokens
                        final_usage.input_tokens = usage_from_context.input_tokens;
                        usage = Some(final_usage);
                    }

                    new_event
                        .json_data(StreamEvent::MessageDelta {
                            delta,
                            usage,
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
        .is_some_and(|v| !v.contains("text-event-stream"))
    {
        resp.extensions_mut().remove::<ClaudeContext>();
    }
    resp
}
