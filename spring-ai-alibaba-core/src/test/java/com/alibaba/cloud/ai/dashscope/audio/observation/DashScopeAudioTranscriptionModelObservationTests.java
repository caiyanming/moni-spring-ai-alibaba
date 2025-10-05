package com.alibaba.cloud.ai.dashscope.audio.observation;

import com.alibaba.cloud.ai.dashscope.api.DashScopeAudioTranscriptionApi;
import com.alibaba.cloud.ai.dashscope.audio.DashScopeAudioTranscriptionModel;
import com.alibaba.cloud.ai.dashscope.audio.DashScopeAudioTranscriptionOptions;
import com.alibaba.cloud.ai.dashscope.common.DashScopeApiConstants;
import com.alibaba.cloud.ai.dashscope.protocol.DashScopeWebSocketClient;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;
import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.ResponseEntity;
import reactor.core.publisher.Flux;
import reactor.test.StepVerifier;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DashScopeAudioTranscriptionModelObservationTests {

	private DashScopeAudioTranscriptionApi api;

	private TestObservationRegistry observationRegistry;

	private Path audioFile;

	@BeforeEach
	void setUp() throws IOException {
		this.api = mock(DashScopeAudioTranscriptionApi.class);
		this.observationRegistry = TestObservationRegistry.create();
		this.audioFile = Files.createTempFile("dashscope-audio", ".wav");
		Files.writeString(this.audioFile, "dummy", StandardCharsets.UTF_8);
	}

	@AfterEach
	void tearDown() throws IOException {
		Files.deleteIfExists(this.audioFile);
	}

	@Test
	void callShouldRecordObservation() {
		DashScopeAudioTranscriptionOptions defaultOptions = DashScopeAudioTranscriptionOptions.builder()
			.withModel("paraformer-test")
			.build();
		DashScopeAudioTranscriptionModel model = new DashScopeAudioTranscriptionModel(this.api, defaultOptions,
				RetryUtils.DEFAULT_RETRY_TEMPLATE, this.observationRegistry);

		AudioTranscriptionPrompt prompt = new AudioTranscriptionPrompt(new FileSystemResource(this.audioFile));

		DashScopeAudioTranscriptionApi.Response.Output.Result result = new DashScopeAudioTranscriptionApi.Response.Output.Result(
				"file-url", "http://example.com/outcome", "SUCCEEDED");
		DashScopeAudioTranscriptionApi.Response.Output submitOutput = new DashScopeAudioTranscriptionApi.Response.Output(
				"task-123", DashScopeAudioTranscriptionApi.TaskStatus.PENDING, null, null, null, List.of(result), null);
		DashScopeAudioTranscriptionApi.Response submitResponse = new DashScopeAudioTranscriptionApi.Response(200,
				"req-123", "OK", null, null, submitOutput);
		when(this.api.call(any(DashScopeAudioTranscriptionApi.Request.class)))
			.thenReturn(ResponseEntity.ok(submitResponse));

		DashScopeAudioTranscriptionApi.Response.Output fetchOutput = new DashScopeAudioTranscriptionApi.Response.Output(
				"task-123", DashScopeAudioTranscriptionApi.TaskStatus.SUCCEEDED, null, null, null, List.of(result),
				null);
		DashScopeAudioTranscriptionApi.Response fetchResponse = new DashScopeAudioTranscriptionApi.Response(200,
				"req-123", "OK", null, null, fetchOutput);
		when(this.api.callWithTaskId(any(DashScopeAudioTranscriptionApi.Request.class), eq("task-123")))
			.thenReturn(ResponseEntity.ok(fetchResponse));

		DashScopeAudioTranscriptionApi.Outcome.Transcript transcript = new DashScopeAudioTranscriptionApi.Outcome.Transcript(
				0, 0, "transcribed text", List.of());
		DashScopeAudioTranscriptionApi.Outcome outcome = new DashScopeAudioTranscriptionApi.Outcome("file-url", null,
				List.of(transcript));
		when(this.api.getOutcome("http://example.com/outcome")).thenReturn(outcome);

		StepVerifier.create(model.call(prompt))
			.assertNext(response -> assertThat(response.getResult().getOutput()).isEqualTo("transcribed text"))
			.verifyComplete();

		TestObservationRegistryAssert.assertThat(this.observationRegistry)
			.hasObservationWithNameEqualTo("gen_ai.client.operation")
			.that()
			.hasContextualNameEqualTo("audio_transcription paraformer-test")
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER.asString(),
					DashScopeApiConstants.PROVIDER_NAME)
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE.asString(),
					"audio_transcription")
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL.asString(),
					"paraformer-test");
	}

	@Test
	void streamShouldRecordObservation() {
		DashScopeAudioTranscriptionOptions defaultOptions = DashScopeAudioTranscriptionOptions.builder()
			.withModel("paraformer-test")
			.build();
		DashScopeAudioTranscriptionModel model = new DashScopeAudioTranscriptionModel(this.api, defaultOptions,
				RetryUtils.DEFAULT_RETRY_TEMPLATE, this.observationRegistry);

		AudioTranscriptionPrompt prompt = new AudioTranscriptionPrompt(
				new ByteArrayResource("audio".getBytes(StandardCharsets.UTF_8)));

		DashScopeAudioTranscriptionApi.RealtimeResponse.Payload.Output.Sentence sentence = new DashScopeAudioTranscriptionApi.RealtimeResponse.Payload.Output.Sentence(
				"sentence-1", 0, 1, "stream text", null, null, true, null, null);
		DashScopeAudioTranscriptionApi.RealtimeResponse.Payload payload = new DashScopeAudioTranscriptionApi.RealtimeResponse.Payload(
				new DashScopeAudioTranscriptionApi.RealtimeResponse.Payload.Output(sentence), null);
		DashScopeAudioTranscriptionApi.RealtimeResponse.Header header = new DashScopeAudioTranscriptionApi.RealtimeResponse.Header(
				"task-456", DashScopeWebSocketClient.EventType.RUN_TASK,
				new DashScopeAudioTranscriptionApi.RealtimeResponse.Header.Attributes());
		DashScopeAudioTranscriptionApi.RealtimeResponse realtimeResponse = new DashScopeAudioTranscriptionApi.RealtimeResponse(
				header, payload);

		when(this.api.realtimeStream(ArgumentMatchers.any(Flux.class))).thenReturn(Flux.just(realtimeResponse));

		StepVerifier.create(model.stream(prompt))
			.assertNext(response -> assertThat(response.getResult().getOutput()).isEqualTo("stream text"))
			.verifyComplete();

		TestObservationRegistryAssert.assertThat(this.observationRegistry)
			.hasObservationWithNameEqualTo("gen_ai.client.operation")
			.that()
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_PROVIDER.asString(),
					DashScopeApiConstants.PROVIDER_NAME)
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.AI_OPERATION_TYPE.asString(),
					"audio_transcription")
			.hasLowCardinalityKeyValue(
					AudioTranscriptionObservationDocumentation.LowCardinalityKeyNames.REQUEST_MODEL.asString(),
					"paraformer-test");
	}

}
