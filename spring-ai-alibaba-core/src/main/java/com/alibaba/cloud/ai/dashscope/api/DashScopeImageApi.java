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

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import static com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants.DEFAULT_BASE_URL;

/**
 * @author nuocheng.lxm
 * @since 1.0.0-M2
 */
public class DashScopeImageApi {

	public static final String DEFAULT_IMAGE_MODEL = ImageModel.WANX_V1.getValue();

	private final WebClient webClient;

	public DashScopeImageApi(String apiKey) {
		this(DEFAULT_BASE_URL, apiKey, WebClient.builder());
	}

	public DashScopeImageApi(String apiKey, String workSpaceId) {
		this(DEFAULT_BASE_URL, apiKey, workSpaceId, WebClient.builder());
	}

	public DashScopeImageApi(String baseUrl, String apiKey, String workSpaceId) {
		this(baseUrl, apiKey, workSpaceId, WebClient.builder());
	}

	public DashScopeImageApi(String baseUrl, String apiKey, WebClient.Builder webClientBuilder) {
		this.webClient = webClientBuilder.baseUrl(baseUrl)
			.defaultHeaders(ApiUtils.getJsonContentHeaders(apiKey, null, true))
			.build();
	}

	public DashScopeImageApi(String baseUrl, String apiKey, String workSpaceId, WebClient.Builder webClientBuilder) {
		this.webClient = webClientBuilder.baseUrl(baseUrl)
			.defaultHeaders(ApiUtils.getJsonContentHeaders(apiKey, workSpaceId, true))
			.build();
	}

	public Mono<DashScopeImageAsyncReponse> submitImageGenTask(DashScopeImageRequest request) {
		String url = "/api/v1/services/aigc/";
		if (request.model().equals("wanx2.1-imageedit") || request.model().equals("wanx-x-painting")
				|| request.model().equals("wanx-sketch-to-image-lite"))
			url += "image2image";
		else
			url += "text2image";
		url += "/image-synthesis";

		return this.webClient.post()
			.uri(url)
			.header("X-DashScope-Async", "enable")
			.body(Mono.just(request), DashScopeImageRequest.class)
			.retrieve()
			.bodyToMono(DashScopeImageAsyncReponse.class);
	}

	public Mono<DashScopeImageAsyncReponse> getImageGenTaskResult(String taskId) {
		return this.webClient.get()
			.uri("/api/v1/tasks/{task_id}", taskId)
			.retrieve()
			.bodyToMono(DashScopeImageAsyncReponse.class);
	}

	/*******************************************
	 * Embedding相关
	 **********************************************/

	public enum ImageModel {

		WANX_V1("wanx-v1");

		public final String value;

		ImageModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return value;
		}

	}

	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record DashScopeImageRequest(@JsonProperty("model") String model,
			@JsonProperty("input") DashScopeImageRequestInput input,
			@JsonProperty("parameters") DashScopeImageRequestParameter parameters

	) {
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageRequestInput(@JsonProperty("prompt") String prompt,
				@JsonProperty("negative_prompt") String negativePrompt, @JsonProperty("ref_img") String refImg,
				@JsonProperty("function") String function, @JsonProperty("base_image_url") String baseImageUrl,
				@JsonProperty("mask_image_url") String maskImageUrl,
				@JsonProperty("sketch_image_url") String sketchImageUrl) {
		}

		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageRequestParameter(@JsonProperty("style") String style,
				@JsonProperty("size") String size, @JsonProperty("n") Integer n, @JsonProperty("seed") Integer seed,
				@JsonProperty("ref_strength") Float refStrength, @JsonProperty("ref_mode") String refMode,
				@JsonProperty("prompt_extend") Boolean promptExtend, @JsonProperty("watermark") Boolean watermark,

				@JsonProperty("sketch_weight") Integer sketchWeight,
				@JsonProperty("sketch_extraction") Boolean sketchExtraction,
				@JsonProperty("sketch_color") Integer[][] sketchColor,
				@JsonProperty("mask_color") Integer[][] maskColor) {
		}
	}

	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record DashScopeImageAsyncReponse(@JsonProperty("request_id") String requestId,
			@JsonProperty("output") DashScopeImageAsyncReponseOutput output,
			@JsonProperty("usage") DashScopeImageAsyncReponseUsage usage) {

		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageAsyncReponseOutput(@JsonProperty("task_id") String taskId,
				@JsonProperty("task_status") String taskStatus,
				@JsonProperty("results") List<DashScopeImageAsyncReponseResult> results,
				@JsonProperty("task_metrics") DashScopeImageAsyncReponseTaskMetrics taskMetrics,
				@JsonProperty("code") String code, @JsonProperty("message") String message) {
		}

		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageAsyncReponseTaskMetrics(@JsonProperty("TOTAL") Integer total,
				@JsonProperty("SUCCEEDED") Integer SUCCEEDED, @JsonProperty("FAILED") Integer FAILED) {
		}

		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageAsyncReponseUsage(@JsonProperty("image_count") Integer imageCount) {
		}

		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record DashScopeImageAsyncReponseResult(@JsonProperty("url") String url) {
		}
	}

}
