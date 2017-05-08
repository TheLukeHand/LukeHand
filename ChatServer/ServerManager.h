/***
* ==++==
*
* Copyright (c) Microsoft Corporation. All rights reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* ServerManager.h
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#pragma once

#include <vector>
#include <memory>
#include <ppltasks.h>
#include <map>
#include <mutex>

#include "..\Common\ServerClientPackets.h"

#include "Poco\Net\HTTPServer.h"
#include "Poco\Net\HTTPRequestHandler.h"
#include "Poco\Net\HTTPRequestHandlerFactory.h"
#include "Poco\Net\HTTPServerRequest.h"
#include "Poco\Net\HTTPServerResponse.h"
#include "Poco\Net\HTTPServerParams.h"
#include "Poco\Net\ServerSocket.h"
#include "Poco\Net\WebSocket.h"
#include "Poco\Net\NetException.h"
#include "Poco\Timespan.h"

enum websocket_message_type
{
	WEB_SOCKET_BINARY_MESSAGE_TYPE,
	WEB_SOCKET_BINARY_FRAGMENT_TYPE,
	WEB_SOCKET_UTF8_MESSAGE_TYPE,
	WEB_SOCKET_UTF8_FRAGMENT_TYPE
};

class WebSocketMessage
{
public:
	WebSocketMessage(unsigned char* buffer, int flags, int n);
	WebSocketMessage(const std::string& msg);

	int get_flags() const;
	const std::vector<unsigned char>& get_data() const;
	websocket_message_type get_message_type() const;
		
	const std::string as_string() const;

private:
	std::vector<unsigned char> m_data;
	websocket_message_type m_msg_type;
};

class ServerManager
{
public:
	ServerManager(const Poco::Net::IPAddress& address, uint16_t port);
	~ServerManager();

	void receive_message(std::shared_ptr<Poco::Net::WebSocket> websocket, const WebSocketMessage& msg);

	void broadcast(const web::json::value& msg);
	void broadcast_pose(uint8_t pose_code);

private:
	void handle_client_login(std::shared_ptr<Poco::Net::WebSocket> websocket, const ServerClientPacket& packet);
	void handle_client_logout(const ServerClientPacket& packet);
	void handle_chat_text_message(const ServerClientPacket& packet);

	void send_client_list(const utility::string_t& client_id);
	void send_message(const utility::string_t& client_id, const WebSocketMessage& msg);

	Poco::Net::ServerSocket m_server_socket;
	Poco::Net::HTTPServer m_http_server;

	std::mutex m_client_info_mutex;
	std::map<utility::string_t, ClientInfo> m_client_list;
	std::map<utility::string_t, std::shared_ptr<Poco::Net::WebSocket>> m_client_websocket_map;
};

class WebSocketRequestHandler : public Poco::Net::HTTPRequestHandler
{
public:
	WebSocketRequestHandler(std::shared_ptr<ServerManager> test_srv);
	void handleRequest(Poco::Net::HTTPServerRequest& request, Poco::Net::HTTPServerResponse& response);

private:
	std::shared_ptr<ServerManager> m_testserver;
};

class RequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
public:
	RequestHandlerFactory(std::shared_ptr<ServerManager> test_srv);
	Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest& request);

private:
	std::shared_ptr<ServerManager> m_testserver;
};
