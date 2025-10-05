package com.alibaba.cloud.ai.dashscope.audio.observation;

import com.alibaba.cloud.ai.dashscope.api.DashScopeSpeechSynthesisApi;
import com.alibaba.cloud.ai.dashscope.audio.DashScopeSpeechSynthesisModel;
import com.alibaba.cloud.ai.dashscope.audio.DashScopeSpeechSynthesisOptions;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisMessage;
import com.alibaba.cloud.ai.dashscope.audio.synthesis.SpeechSynthesisPrompt;
import com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.retry.RetryUtils;
import reactor.core.publisher.Flux;
import reactor.test.StepVerifier;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DashScopeSpeechSynthesisModelObservationTests {

	private DashScopeSpeechSynthesisApi api;

	private TestObservationRegistry observationRegistry;

	@BeforeEach
	void setUp() {
		this.api = mock(DashScopeSpeechSynthesisApi.class);
		this.observationRegistry = TestObservationRegistry.create();
	}

	@Test
	void callShouldRecordObservation() {
		DashScopeSpeechSynthesisOptions defaultOptions = DashScopeSpeechSynthesisOptions.builder()
			.model("cosyvoice-test")
			.build();
		DashScopeSpeechSynthesisModel model = new DashScopeSpeechSynthesisModel(this.api, defaultOptions,
				RetryUtils.DEFAULT_RETRY_TEMPLATE, this.observationRegistry);

		SpeechSynthesisPrompt prompt = new SpeechSynthesisPrompt(new SpeechSynthesisMessage("你好，世界"));

		ByteBuffer audioBuffer = ByteBuffer.wrap("audio-chunk".getBytes(StandardCharsets.UTF_8));
		when(this.api.streamOut(any(DashScopeSpeechSynthesisApi.Request.class))).thenReturn(Flux.just(audioBuffer));

		StepVerifier.create(model.call(prompt)).assertNext(response -> {
			assertThat(response).isNotNull();
			ByteBuffer buffer = Objects.requireNonNull(response.getResult().getOutput().getAudio());
			assertThat(buffer.capacity()).isEqualTo(audioBuffer.capacity());
		}).verifyComplete();

		TestObservationRegistryAssert.assertThat(this.observationRegistry)
			.hasObservationWithNameEqualTo("gen_ai.client.operation")
			.that()
			.hasContextualNameEqualTo("speech_synthesis cosyvoice-test")
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE.asString(),
					"speech_synthesis")
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER.asString(),
					DashScopeApiConstants.PROVIDER_NAME)
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL.asString(),
					"cosyvoice-test");
	}

	@Test
	void streamShouldRecordObservation() {
		DashScopeSpeechSynthesisOptions defaultOptions = DashScopeSpeechSynthesisOptions.builder()
			.model("cosyvoice-test")
			.build();
		DashScopeSpeechSynthesisModel model = new DashScopeSpeechSynthesisModel(this.api, defaultOptions,
				RetryUtils.DEFAULT_RETRY_TEMPLATE, this.observationRegistry);

		SpeechSynthesisPrompt prompt = new SpeechSynthesisPrompt(new SpeechSynthesisMessage("hello"));

		when(this.api.streamOut(any(DashScopeSpeechSynthesisApi.Request.class)))
			.thenReturn(Flux.just(ByteBuffer.wrap("chunk-1".getBytes(StandardCharsets.UTF_8))));

		StepVerifier.create(model.stream(prompt))
			.assertNext(
					response -> assertThat(response.getResult().getOutput().getAudio().remaining()).isGreaterThan(0))
			.verifyComplete();

		TestObservationRegistryAssert.assertThat(this.observationRegistry)
			.hasObservationWithNameEqualTo("gen_ai.client.operation")
			.that()
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE.asString(),
					"speech_synthesis")
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER.asString(),
					DashScopeApiConstants.PROVIDER_NAME)
			.hasLowCardinalityKeyValue(
					SpeechSynthesisObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL.asString(),
					"cosyvoice-test");
	}

}
