/*
 * Copyright 2024-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.cloud.ai.dashscope.api;

import com.alibaba.cloud.ai.dashscope.api.DashScopeApi.*;
import com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.model.ApiKey;
import org.springframework.ai.model.NoopApiKey;
import org.springframework.ai.model.SimpleApiKey;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * Pure reactive DashScope API client using WebClient for all operations. This provides
 * truly non-blocking, reactive HTTP calls without any Mono.fromCallable() +
 * boundedElastic workarounds.
 *
 * @author Claude (MoniAI)
 * @since 1.0.0-M5.1
 */
public class ReactiveDashScopeApi {

	private static final Logger logger = LoggerFactory.getLogger(ReactiveDashScopeApi.class);

	private static final Predicate<String> SSE_DONE_PREDICATE = "[DONE]"::equals;

	// Configuration fields
	private final String baseUrl;

	private final ApiKey apiKey;

	private final String completionsPath;

	private final String embeddingsPath;

	private final MultiValueMap<String, String> headers;

	private final WebClient webClient;

	private final ObjectMapper objectMapper;

	/**
	 * Default chat model
	 */
	public static final String DEFAULT_CHAT_MODEL = DashScopeApi.ChatModel.QWEN_PLUS.getValue();

	public static final String DEFAULT_EMBEDDING_MODEL = DashScopeApi.EmbeddingModel.EMBEDDING_V2.getValue();

	/**
	 * Create a new ReactiveDashScopeApi instance.
	 */
	public ReactiveDashScopeApi(String baseUrl, String apiKey, WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper) {
		this(baseUrl, new SimpleApiKey(apiKey), webClientBuilder, objectMapper, null);
	}

	/**
	 * Create a new ReactiveDashScopeApi instance with custom headers.
	 */
	public ReactiveDashScopeApi(String baseUrl, ApiKey apiKey, WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper, MultiValueMap<String, String> headers) {
		this.baseUrl = baseUrl;
		this.apiKey = apiKey;
		this.completionsPath = "/api/v1/services/aigc/text-generation/generation";
		this.embeddingsPath = "/api/v1/services/embeddings/text-embedding/text-embedding";
		this.headers = headers;
		this.objectMapper = objectMapper;

		// Build WebClient with base URL and default headers
		this.webClient = webClientBuilder.baseUrl(baseUrl)
			.defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
			.defaultHeader("Accept", MediaType.APPLICATION_JSON_VALUE)
			.defaultHeader("User-Agent", "spring-ai-alibaba/1.0.0")
			.build();
	}

	/**
	 * Builder for ReactiveDashScopeApi
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Pure reactive chat completion - no blocking calls!
	 */
	public Mono<ChatCompletion> chatCompletion(ChatCompletionRequest chatRequest) {
		return chatCompletion(chatRequest, null);
	}

	/**
	 * Pure reactive chat completion with additional headers
	 */
	public Mono<ChatCompletion> chatCompletion(ChatCompletionRequest chatRequest,
			MultiValueMap<String, String> additionalHeaders) {

		if (chatRequest == null) {
			return Mono.error(new IllegalArgumentException("The request body cannot be null"));
		}

		if (chatRequest.stream()) {
			return Mono.error(new IllegalArgumentException("Request must set the stream property to false"));
		}

		String chatCompletionUri = chatRequest.multiModel() ? "/api/v1/services/aigc/multimodal-generation/generation"
				: this.completionsPath;

		return this.webClient.post().uri(chatCompletionUri).headers(headers -> {
			if (additionalHeaders != null) {
				headers.addAll(additionalHeaders);
			}
			addDefaultHeadersIfMissing(headers);
		})
			.bodyValue(chatRequest)
			.retrieve()
			.onStatus(status -> status.isError(), response -> response.bodyToMono(String.class)
				.map(body -> new RuntimeException("DashScope API error: " + response.statusCode() + " - " + body)))
			.bodyToMono(ChatCompletion.class)
			.retryWhen(Retry.backoff(3, Duration.ofSeconds(1)).filter(this::isRetryableError))
			.doOnError(error -> logger.error("Chat completion failed", error))
			.doOnSuccess(response -> logger.debug("Chat completion successful"));
	}

	/**
	 * Pure reactive streaming chat completion
	 */
	public Flux<ChatCompletionChunk> chatCompletionStream(ChatCompletionRequest chatRequest) {
		return chatCompletionStream(chatRequest, null);
	}

	/**
	 * Pure reactive streaming chat completion with additional headers
	 */
	public Flux<ChatCompletionChunk> chatCompletionStream(ChatCompletionRequest chatRequest,
			MultiValueMap<String, String> additionalHeaders) {

		if (chatRequest == null) {
			return Flux.error(new IllegalArgumentException("The request body cannot be null"));
		}

		if (!chatRequest.stream()) {
			return Flux.error(new IllegalArgumentException("Request must set the stream property to true"));
		}

		String chatCompletionUri = chatRequest.multiModel() ? "/api/v1/services/aigc/multimodal-generation/generation"
				: this.completionsPath;

		return this.webClient.post().uri(chatCompletionUri).headers(headers -> {
			if (additionalHeaders != null) {
				headers.addAll(additionalHeaders);
			}
			// For DashScope stream
			headers.add("X-DashScope-SSE", "enable");
			addDefaultHeadersIfMissing(headers);
		})
			.bodyValue(chatRequest)
			.retrieve()
			.onStatus(status -> status.isError(),
					response -> response.bodyToMono(String.class)
						.map(body -> new RuntimeException(
								"DashScope streaming API error: " + response.statusCode() + " - " + body)))
			.bodyToFlux(String.class)
			.filter(chunk -> StringUtils.hasText(chunk) && !SSE_DONE_PREDICATE.test(chunk.trim()))
			.map(this::parseStreamingChunk)
			.filter(Objects::nonNull)
			.retryWhen(Retry.backoff(3, Duration.ofSeconds(1)).filter(this::isRetryableError))
			.doOnError(error -> logger.error("Streaming chat completion failed", error));
	}

	/**
	 * Pure reactive embeddings - no blocking calls!
	 */
	public Mono<EmbeddingList> embeddings(EmbeddingRequest embeddingRequest) {
		if (embeddingRequest == null) {
			return Mono.error(new IllegalArgumentException("The request body cannot be null"));
		}

		return this.webClient.post()
			.uri(this.embeddingsPath)
			.headers(this::addDefaultHeadersIfMissing)
			.bodyValue(embeddingRequest)
			.retrieve()
			.onStatus(status -> status.isError(),
					response -> response.bodyToMono(String.class)
						.map(body -> new RuntimeException(
								"DashScope embeddings API error: " + response.statusCode() + " - " + body)))
			.bodyToMono(EmbeddingList.class)
			.retryWhen(Retry.backoff(3, Duration.ofSeconds(1)).filter(this::isRetryableError))
			.doOnError(error -> logger.error("Embeddings failed", error))
			.doOnSuccess(response -> logger.debug("Embeddings successful"));
	}

	/**
	 * Add default headers if missing
	 */
	private void addDefaultHeadersIfMissing(org.springframework.http.HttpHeaders headers) {
		if (!(this.apiKey instanceof NoopApiKey)) {
			headers.setBearerAuth(this.apiKey.getValue());
		}

		if (this.headers != null) {
			this.headers.forEach((key, values) -> {
				if (!headers.containsKey(key)) {
					headers.addAll(key, values);
				}
			});
		}
	}

	/**
	 * Parse streaming SSE chunk to ChatCompletionChunk
	 */
	private ChatCompletionChunk parseStreamingChunk(String chunk) {
		try {
			// Remove "data: " prefix if present
			String jsonChunk = chunk.startsWith("data: ") ? chunk.substring(6) : chunk;
			return this.objectMapper.readValue(jsonChunk, ChatCompletionChunk.class);
		}
		catch (JsonProcessingException e) {
			logger.debug("Failed to parse streaming chunk: {}", chunk, e);
			return null;
		}
	}

	/**
	 * Determine if an error is retryable
	 */
	private boolean isRetryableError(Throwable throwable) {
		if (throwable instanceof WebClientResponseException webEx) {
			HttpStatus status = HttpStatus.resolve(webEx.getStatusCode().value());
			// Retry on 5xx server errors and 429 rate limiting
			return status != null && (status.is5xxServerError() || status == HttpStatus.TOO_MANY_REQUESTS);
		}
		// Retry on network errors
		return throwable instanceof java.net.ConnectException || throwable instanceof java.net.SocketTimeoutException;
	}

	/**
	 * Builder class for ReactiveDashScopeApi
	 */
	public static class Builder {

		private String baseUrl = DashScopeApiConstants.DEFAULT_BASE_URL;

		private ApiKey apiKey = new NoopApiKey();

		private WebClient.Builder webClientBuilder = WebClient.builder();

		private ObjectMapper objectMapper = new ObjectMapper();

		private MultiValueMap<String, String> headers;

		public Builder baseUrl(String baseUrl) {
			this.baseUrl = baseUrl;
			return this;
		}

		public Builder apiKey(String apiKey) {
			this.apiKey = new SimpleApiKey(apiKey);
			return this;
		}

		public Builder apiKey(ApiKey apiKey) {
			this.apiKey = apiKey;
			return this;
		}

		public Builder webClientBuilder(WebClient.Builder webClientBuilder) {
			this.webClientBuilder = webClientBuilder;
			return this;
		}

		public Builder objectMapper(ObjectMapper objectMapper) {
			this.objectMapper = objectMapper;
			return this;
		}

		public Builder headers(MultiValueMap<String, String> headers) {
			this.headers = headers;
			return this;
		}

		public ReactiveDashScopeApi build() {
			return new ReactiveDashScopeApi(baseUrl, apiKey, webClientBuilder, objectMapper, headers);
		}

	}

}
