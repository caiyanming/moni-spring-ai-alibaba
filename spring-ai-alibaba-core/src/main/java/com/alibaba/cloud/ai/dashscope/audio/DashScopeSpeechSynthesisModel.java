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
package com.alibaba.cloud.ai.dashscope.audio;

import java.nio.ByteBuffer;
import java.util.UUID;

import com.alibaba.cloud.ai.dashscope.api.DashScopeSpeechSynthesisApi;
import com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants;
import com.alibaba.cloud.ai.dashscope.audio.observation.DefaultSpeechSynthesisObservationConvention;
import com.alibaba.cloud.ai.dashscope.audio.observation.SpeechSynthesisObservationContext;
import com.alibaba.cloud.ai.dashscope.audio.observation.SpeechSynthesisObservationConvention;
import com.alibaba.cloud.ai.dashscope.audio.observation.SpeechSynthesisObservationDocumentation;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisModel;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisOptions;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisOutput;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisPrompt;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisResponse;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import io.micrometer.observation.ObservationRegistry;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * @author kevinlin09
 */
public class DashScopeSpeechSynthesisModel implements SpeechSynthesisModel {

	private static final Logger logger = LoggerFactory.getLogger(DashScopeSpeechSynthesisModel.class);

	private static final SpeechSynthesisObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultSpeechSynthesisObservationConvention();

	private final DashScopeSpeechSynthesisApi api;

	private final DashScopeSpeechSynthesisOptions options;

	private final RetryTemplate retryTemplate;

	private final ObservationRegistry observationRegistry;

	private SpeechSynthesisObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public DashScopeSpeechSynthesisModel(DashScopeSpeechSynthesisApi api) {
		this(api, DashScopeSpeechSynthesisOptions.builder().model("").build());
	}

	public DashScopeSpeechSynthesisModel(DashScopeSpeechSynthesisApi api, DashScopeSpeechSynthesisOptions options) {
		this(api, options, RetryUtils.DEFAULT_RETRY_TEMPLATE, ObservationRegistry.NOOP);
	}

	public DashScopeSpeechSynthesisModel(DashScopeSpeechSynthesisApi api, DashScopeSpeechSynthesisOptions options,
			RetryTemplate retryTemplate) {
		this(api, options, retryTemplate, ObservationRegistry.NOOP);
	}

	public DashScopeSpeechSynthesisModel(DashScopeSpeechSynthesisApi api, DashScopeSpeechSynthesisOptions options,
			RetryTemplate retryTemplate, ObservationRegistry observationRegistry) {
		Assert.notNull(api, "DashScopeSpeechSynthesisApi must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");
		this.api = api;
		this.options = options;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry != null ? observationRegistry : ObservationRegistry.NOOP;
	}

	public enum DashScopeSpeechModel {

		SAMBERT_ZHICHU_V1("sambert-zhichu-v1"),

		COSYVOICE_V1("cosyvoice-v1");

		private final String model;

		DashScopeSpeechModel(String model) {
			this.model = model;
		}

		public String getModel() {
			return this.model;
		}

	}

	@Override
	public Mono<SpeechSynthesisResponse> call(SpeechSynthesisPrompt prompt) {
		return observeMono(prompt, () -> streamInternal(prompt).reduce(this::mergeResponses));
	}

	@Override
	public Flux<SpeechSynthesisResponse> stream(SpeechSynthesisPrompt prompt) {
		return observeFlux(prompt, () -> streamInternal(prompt));
	}

	private SpeechSynthesisObservationContext createObservationContext(SpeechSynthesisPrompt prompt) {
		DashScopeSpeechSynthesisOptions effectiveOptions = resolveOptions(prompt);
		SpeechSynthesisPrompt observationPrompt = new SpeechSynthesisPrompt(prompt.getInstructions(), effectiveOptions);
		return SpeechSynthesisObservationContext.builder()
			.prompt(observationPrompt)
			.provider(DashScopeApiConstants.PROVIDER_NAME)
			.build();
	}

	private void setResponseIfPresent(SpeechSynthesisObservationContext observationContext,
			SpeechSynthesisResponse response) {
		if (response != null) {
			observationContext.setResponse(response);
		}
	}

	private Mono<SpeechSynthesisResponse> observeMono(SpeechSynthesisPrompt prompt,
			Supplier<Mono<SpeechSynthesisResponse>> supplier) {
		SpeechSynthesisObservationContext observationContext = createObservationContext(prompt);
		return SpeechSynthesisObservationDocumentation.SPEECH_SYNTHESIS_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				Mono<SpeechSynthesisResponse> result = supplier.get();
				return result.doOnNext(response -> setResponseIfPresent(observationContext, response));
			});
	}

	private Flux<SpeechSynthesisResponse> observeFlux(SpeechSynthesisPrompt prompt,
			Supplier<Flux<SpeechSynthesisResponse>> supplier) {
		SpeechSynthesisObservationContext observationContext = createObservationContext(prompt);
		return SpeechSynthesisObservationDocumentation.SPEECH_SYNTHESIS_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				Flux<SpeechSynthesisResponse> result = supplier.get();
				return result.doOnNext(response -> setResponseIfPresent(observationContext, response));
			});
	}

	private Flux<SpeechSynthesisResponse> streamInternal(SpeechSynthesisPrompt prompt) {
		return this.retryTemplate.execute(ctx -> this.api.streamOut(createRequest(prompt))
			.map(SpeechSynthesisOutput::new)
			.map(SpeechSynthesisResult::new)
			.map(SpeechSynthesisResponse::new));
	}

	private SpeechSynthesisResponse mergeResponses(SpeechSynthesisResponse first, SpeechSynthesisResponse second) {
		ByteBuffer firstBuffer = first.getResult().getOutput().getAudio();
		ByteBuffer secondBuffer = second.getResult().getOutput().getAudio();
		ByteBuffer combinedBuffer = ByteBuffer.allocate(firstBuffer.remaining() + secondBuffer.remaining());
		combinedBuffer.put(firstBuffer.duplicate());
		combinedBuffer.put(secondBuffer.duplicate());
		combinedBuffer.flip();
		return new SpeechSynthesisResponse(new SpeechSynthesisResult(new SpeechSynthesisOutput(combinedBuffer)));
	}

	public void setObservationConvention(SpeechSynthesisObservationConvention observationConvention) {
		this.observationConvention = Objects.requireNonNull(observationConvention,
				"observationConvention cannot be null");
	}

	public DashScopeSpeechSynthesisApi.Request createRequest(SpeechSynthesisPrompt prompt) {
		DashScopeSpeechSynthesisOptions options = resolveOptions(prompt);

		return new DashScopeSpeechSynthesisApi.Request(
				new DashScopeSpeechSynthesisApi.Request.RequestHeader("run-task", UUID.randomUUID().toString(), "out"),
				new DashScopeSpeechSynthesisApi.Request.RequestPayload(options.getModel(), "audio", "tts",
						"SpeechSynthesizer",
						new DashScopeSpeechSynthesisApi.Request.RequestPayload.RequestPayloadInput(
								prompt.getInstructions().get(0).getText()),
						new DashScopeSpeechSynthesisApi.Request.RequestPayload.RequestPayloadParameters(
								options.getVolume(), options.getRequestTextType().getValue(), options.getVoice(),
								options.getSampleRate(), options.getSpeed(), options.getResponseFormat().getValue(),
								options.getPitch(), options.getEnablePhonemeTimestamp(),
								options.getEnableWordTimestamp())));
	}

	private DashScopeSpeechSynthesisOptions resolveOptions(SpeechSynthesisPrompt prompt) {
		DashScopeSpeechSynthesisOptions options = DashScopeSpeechSynthesisOptions.builder().build();
		if (prompt.getOptions() != null) {
			DashScopeSpeechSynthesisOptions runtimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(),
					SpeechSynthesisOptions.class, DashScopeSpeechSynthesisOptions.class);
			options = ModelOptionsUtils.merge(runtimeOptions, options, DashScopeSpeechSynthesisOptions.class);
		}
		return ModelOptionsUtils.merge(this.options, options, DashScopeSpeechSynthesisOptions.class);
	}

	private SpeechSynthesisResponse toResponse(DashScopeSpeechSynthesisApi.Response apiResponse) {
		SpeechSynthesisOutput output = new SpeechSynthesisOutput(apiResponse.getAudio());
		SpeechSynthesisResult result = new SpeechSynthesisResult(output);
		return new SpeechSynthesisResponse(result);
	}

}
