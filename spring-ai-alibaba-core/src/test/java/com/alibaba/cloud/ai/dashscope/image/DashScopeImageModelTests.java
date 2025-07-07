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
package com.alibaba.cloud.ai.dashscope.image;

import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi;
import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi.DashScopeImageAsyncReponse;
import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi.DashScopeImageAsyncReponse.DashScopeImageAsyncReponseOutput;
import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi.DashScopeImageAsyncReponse.DashScopeImageAsyncReponseResult;
import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi.DashScopeImageAsyncReponse.DashScopeImageAsyncReponseUsage;
import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.image.ImageResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * Test cases for DashScopeImageModel. Tests cover basic image generation, custom options,
 * async task handling, error handling, and edge cases.
 *
 * @author yuluo
 * @author polaris
 * @author brianxiadong
 * @since 1.0.0-M5.1
 */
class DashScopeImageModelTests {

	// Test constants
	private static final String TEST_MODEL = "wanx-v1";

	private static final String TEST_TASK_ID = "test-task-id";

	private static final String TEST_REQUEST_ID = "test-request-id";

	private static final String TEST_IMAGE_URL = "https://example.com/image.jpg";

	private static final String TEST_PROMPT = "A beautiful sunset over mountains";

	private DashScopeImageApi dashScopeImageApi;

	private DashScopeImageModel imageModel;

	private DashScopeImageOptions defaultOptions;

	@BeforeEach
	void setUp() {
		// Initialize mock objects and test instances
		dashScopeImageApi = Mockito.mock(DashScopeImageApi.class);
		defaultOptions = DashScopeImageOptions.builder().withModel(TEST_MODEL).withN(1).build();
		imageModel = new DashScopeImageModel(dashScopeImageApi, defaultOptions, ObservationRegistry.NOOP);
	}

	@Test
	void testBasicImageGeneration() {
		// Test basic reactive image generation with successful response
		mockSuccessfulImageGeneration();

		ImagePrompt prompt = new ImagePrompt(TEST_PROMPT);
		Mono<ImageResponse> responseMono = imageModel.callReactive(prompt);

		StepVerifier.create(responseMono).assertNext(response -> {
			assertThat(response.getResults()).hasSize(1);
			assertThat(response.getResult().getOutput().getUrl()).isEqualTo(TEST_IMAGE_URL);
		}).verifyComplete();
	}

	@Test
	void testCustomOptions() {
		// Test reactive image generation with custom options
		mockSuccessfulImageGeneration();

		DashScopeImageOptions customOptions = DashScopeImageOptions.builder()
			.withModel(TEST_MODEL)
			.withN(2)
			.withWidth(1024)
			.withHeight(1024)
			.withStyle("photography")
			.withSeed(42)
			.build();

		ImagePrompt prompt = new ImagePrompt(TEST_PROMPT, customOptions);
		Mono<ImageResponse> responseMono = imageModel.callReactive(prompt);

		StepVerifier.create(responseMono).assertNext(response -> {
			assertThat(response.getResults()).hasSize(1);
			assertThat(response.getResult().getOutput().getUrl()).isEqualTo(TEST_IMAGE_URL);
		}).verifyComplete();
	}

	@Test
	void testFailedImageGeneration() {
		// Test handling of failed reactive image generation
		mockFailedImageGeneration();

		ImagePrompt prompt = new ImagePrompt(TEST_PROMPT);
		Mono<ImageResponse> responseMono = imageModel.callReactive(prompt);

		StepVerifier.create(responseMono)
			.assertNext(response -> assertThat(response.getResults()).isEmpty())
			.verifyComplete();
	}

	@Test
	void testNullResponse() {
		// Test handling of null API response reactively
		// When submitImageGenTask returns empty, it means no task ID was generated
		when(dashScopeImageApi.submitImageGenTask(any())).thenReturn(Mono.empty());

		ImagePrompt prompt = new ImagePrompt(TEST_PROMPT);
		Mono<ImageResponse> responseMono = imageModel.callReactive(prompt);

		// When there's no task ID, the reactive chain should return an empty response
		StepVerifier.create(responseMono)
			.assertNext(response -> {
				assertThat(response.getResults()).isEmpty();
				assertThat((String) response.getMetadata().get("taskStatus")).isEqualTo("NO_TASK_ID");
			})
			.verifyComplete();
	}

	@Test
	void testNullPrompt() {
		// Test handling of null prompt reactively
		assertThatThrownBy(() -> imageModel.callReactive(null)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("Prompt");
	}

	@Test
	void testEmptyPrompt() {
		// Test handling of empty prompt reactively
		assertThatThrownBy(() -> imageModel.callReactive(new ImagePrompt(new ArrayList<>())))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("Prompt");
	}

	@Test
	void testReactiveApiIsPreferred() {
		// Test that the reactive API is the preferred interface
		mockSuccessfulImageGeneration();

		ImagePrompt prompt = new ImagePrompt(TEST_PROMPT);

		// Verify reactive method exists and works
		Mono<ImageResponse> responseMono = imageModel.callReactive(prompt);
		assertThat(responseMono).isNotNull();

		// Verify we can use reactive patterns
		StepVerifier.create(responseMono.map(response -> response.getResults().size()).filter(size -> size > 0))
			.expectNext(1)
			.verifyComplete();
	}

	private void mockSuccessfulImageGeneration() {
		// Mock successful task submission
		DashScopeImageAsyncReponse submitResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "PENDING", null, null, null, null),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.submitImageGenTask(any())).thenReturn(Mono.just(submitResponse));

		// Mock successful task completion
		DashScopeImageAsyncReponse completedResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "SUCCEEDED",
						List.of(new DashScopeImageAsyncReponseResult(TEST_IMAGE_URL)), null, null, null),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.getImageGenTaskResult(TEST_TASK_ID)).thenReturn(Mono.just(completedResponse));
	}

	private void mockFailedImageGeneration() {
		// Mock successful task submission but failed completion
		DashScopeImageAsyncReponse submitResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "PENDING", null, null, null, null),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.submitImageGenTask(any())).thenReturn(Mono.just(submitResponse));

		// Mock failed task completion
		DashScopeImageAsyncReponse failedResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "FAILED", null, null, "ERROR_CODE", "Error message"),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.getImageGenTaskResult(TEST_TASK_ID)).thenReturn(Mono.just(failedResponse));
	}

	private void mockTimeoutImageGeneration() {
		// Mock successful task submission but pending status until timeout
		DashScopeImageAsyncReponse submitResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "PENDING", null, null, null, null),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.submitImageGenTask(any())).thenReturn(Mono.just(submitResponse));

		// Mock pending status for all status checks
		DashScopeImageAsyncReponse pendingResponse = new DashScopeImageAsyncReponse(TEST_REQUEST_ID,
				new DashScopeImageAsyncReponseOutput(TEST_TASK_ID, "PENDING", null, null, null, null),
				new DashScopeImageAsyncReponseUsage(1));
		when(dashScopeImageApi.getImageGenTaskResult(TEST_TASK_ID)).thenReturn(Mono.just(pendingResponse));
	}

}
