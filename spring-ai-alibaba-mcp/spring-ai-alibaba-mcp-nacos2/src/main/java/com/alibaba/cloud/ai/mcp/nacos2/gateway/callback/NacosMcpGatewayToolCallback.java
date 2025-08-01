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

package com.alibaba.cloud.ai.mcp.nacos2.gateway.callback;

import com.alibaba.cloud.ai.mcp.nacos2.gateway.definition.NacosMcpGatewayToolDefinition;
import com.alibaba.cloud.ai.mcp.nacos2.gateway.utils.SpringBeanUtils;
import com.alibaba.nacos.api.exception.NacosException;
import com.alibaba.nacos.api.naming.NamingService;
import com.alibaba.nacos.api.naming.pojo.Instance;
import com.alibaba.nacos.common.utils.JacksonUtils;
import com.alibaba.nacos.shaded.com.google.common.collect.Maps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.lang.NonNull;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

public class NacosMcpGatewayToolCallback implements ToolCallback {

	private static final Logger logger = LoggerFactory.getLogger(NacosMcpGatewayToolCallback.class);

	private final ToolDefinition toolDefinition;

	private final WebClient webClient;

	private final NamingService namingService;

	public NacosMcpGatewayToolCallback(final ToolDefinition toolDefinition) {
		this.toolDefinition = toolDefinition;
		this.webClient = SpringBeanUtils.getInstance().getBean(WebClient.class);
		this.namingService = SpringBeanUtils.getInstance().getBean(NamingService.class);
	}

	@Override
	public ToolDefinition getToolDefinition() {
		return this.toolDefinition;
	}

	@Override
	public Mono<String> call(@NonNull final String input) {
		return call(input, new ToolContext(Maps.newHashMap()));
	}

	@Override
	public Mono<String> call(@NonNull final String input, final ToolContext toolContext) {
		return Mono.fromCallable(() -> {
			logger.info("Tool callback: {} input: {}, toolContext: {}", toolDefinition.name(), input, toolContext);
			NacosMcpGatewayToolDefinition nacosToolDefinition = (NacosMcpGatewayToolDefinition) this.toolDefinition;
			logger.info("Tool callback toolDefinition: {}", JacksonUtils.toJson(nacosToolDefinition));
			try {
				Instance instance = namingService.selectOneHealthyInstance(nacosToolDefinition.getServiceName());
				logger.info("Tool callback instance: {}", JacksonUtils.toJson(instance));
				String url = "http://" + instance.getIp() + ":" + instance.getPort()
						+ nacosToolDefinition.getRequestPath();
				logger.info("Tool callback url: {}", url);
				return url;
			}
			catch (NacosException e) {
				throw new RuntimeException(e);
			}
		}).flatMap(url -> {
			NacosMcpGatewayToolDefinition nacosToolDefinition = (NacosMcpGatewayToolDefinition) this.toolDefinition;
			if (nacosToolDefinition.getRequestMethod().equalsIgnoreCase("POST")) {
				return webClient.post().uri(url).bodyValue(input).retrieve().bodyToMono(String.class);
			}
			return webClient.get().uri(url).retrieve().bodyToMono(String.class);
		});
	}

}
