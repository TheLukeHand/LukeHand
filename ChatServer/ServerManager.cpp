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
* ServerManager.cpp 
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/

#include "stdafx.h"
#include "cpprest/asyncrt_utils.h"
#include "ServerManager.h"

using namespace Poco::Net;
using namespace utility;
using namespace utility::conversions;
using namespace web;

#pragma region Webcocket_message

// The data buffer received by the server has unnecessary extra character
// beyond the reported number of bytes received. This classes implementation 
// has some redundant copy of data to circumvent this while keeping implementation
// simple for sample purposes.

WebSocketMessage::WebSocketMessage(unsigned char* buffer, int flags, int n) : m_data(buffer, buffer + n)
{
	switch (flags)
	{
	case WebSocket::FRAME_TEXT:
		m_msg_type = websocket_message_type::WEB_SOCKET_UTF8_MESSAGE_TYPE;
		break;
	case WebSocket::FRAME_BINARY:
		m_msg_type = websocket_message_type::WEB_SOCKET_BINARY_MESSAGE_TYPE;
		break;
	}
}

WebSocketMessage::WebSocketMessage(const std::string& msg) : m_data(msg.begin(), msg.end())
{
	m_msg_type = websocket_message_type::WEB_SOCKET_UTF8_MESSAGE_TYPE;
}

const std::vector<unsigned char>& WebSocketMessage::get_data() const
{
	return m_data;
}

websocket_message_type WebSocketMessage::get_message_type()const
{
	return m_msg_type;
}

int WebSocketMessage::get_flags() const 
{
	int flags = 0;

	switch (m_msg_type)
	{
	case websocket_message_type::WEB_SOCKET_UTF8_FRAGMENT_TYPE:
		flags = WebSocket::FRAME_OP_CONT | WebSocket::FRAME_OP_TEXT;
		break;
	case websocket_message_type::WEB_SOCKET_UTF8_MESSAGE_TYPE:
		flags = WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_TEXT;
		break;
	case websocket_message_type::WEB_SOCKET_BINARY_FRAGMENT_TYPE:
		flags = WebSocket::FRAME_OP_CONT | WebSocket::FRAME_OP_BINARY;
		break;
	case websocket_message_type::WEB_SOCKET_BINARY_MESSAGE_TYPE:
		flags = WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_BINARY;
		break;
	}

	return flags;
}

const std::string WebSocketMessage::as_string() const
{
	std::string temp(m_data.begin(), m_data.end());
	return temp;
}

#pragma endregion

#pragma region WebsocketServer

ServerManager::ServerManager(const IPAddress& address, uint16_t port) :
	m_server_socket(SocketAddress(address, port)),
	m_http_server(new RequestHandlerFactory(std::shared_ptr<ServerManager>(this)), m_server_socket, new HTTPServerParams())
{
	m_http_server.start();

	std::cout << std::endl << "======================================" << std::endl << std::endl;

	std::cout << "Running server at IP: " << m_server_socket.address().toString() << std::endl;

	std::cout << std::endl << "======================================" << std::endl << std::endl;
}

ServerManager::~ServerManager()
{
	m_client_list.clear();
	m_client_websocket_map.clear();
	m_http_server.stopAll();
	m_server_socket.close();
}

void ServerManager::send_message(const utility::string_t& client_id, const WebSocketMessage& msg)
{	
	auto ws_iter = m_client_websocket_map.find(client_id);

	if (ws_iter == m_client_websocket_map.end())
	{
		// If websocket is missing simply ignore the message.
		std::wcout << "Websocket missing for client id: " << client_id << std::endl;
		return;
	}

	std::shared_ptr<WebSocket> websocket_ptr = ws_iter->second;
	
	const auto& data = msg.get_data();
	int flags = msg.get_flags();

	if (data.size() == 0)
	{
		websocket_ptr->sendFrame(nullptr, static_cast<int>(data.size()), flags);
	}
	else
	{
		websocket_ptr->sendFrame(&data[0], static_cast<int>(data.size()), flags);
	}
}

// Rest = 0, Fist = 1, WaveIn = 2, WaveOut = 3, FingersSpread = 4, DoubleTap = 5
void ServerManager::broadcast_pose(uint8_t pose_code) {
	json::value Data;
	Data[U("Pose")] = web::json::value::number(pose_code);

	json::value msg;
	msg[U("Data")] = Data;

	broadcast(msg);
}

void ServerManager::receive_message(std::shared_ptr<WebSocket> websocket, const WebSocketMessage& msg)
{
	websocket_message_type msg_type = msg.get_message_type();
	if (msg_type == websocket_message_type::WEB_SOCKET_UTF8_MESSAGE_TYPE)
	{
		auto msg_str = msg.as_string();
		if (msg_str == "<BEGIN>Connected<END>") {
			auto address = to_string_t(websocket->address().toString());
			m_client_websocket_map[address] = websocket;
		}
		else if (msg_str == "<BEGIN>Close<END>") {
			auto address = to_string_t(websocket->address().toString());
			m_client_websocket_map.erase(address); // TODO check
		}
		else {
			json::value json_value = json::value::parse(to_string_t(msg.as_string()));

			ServerClientPacket packet(json_value);
			packet_content_type content_type = packet.get_content_type();

			if (content_type == packet_content_type::CLIENT_LOGIN)
			{
				this->handle_client_login(websocket, packet);
			}
			else if (content_type == packet_content_type::CLIENT_LOGOUT)
			{
				this->handle_client_logout(packet);
			}
			else if (content_type == packet_content_type::CHAT_TEXT_MSG)
			{
				this->handle_chat_text_message(packet);
			}
		}
	}
}

void ServerManager::broadcast(const web::json::value& msg) {
	WebSocketMessage out_msg(to_utf8string(msg.serialize()));
	for (const auto& id_socket : m_client_websocket_map) {
		send_message(id_socket.first, out_msg);
	}
}

void ServerManager::handle_client_login(std::shared_ptr<WebSocket> websocket, const ServerClientPacket& packet)
{
	const ClientInfo& c_info = packet.get_local_client();

	std::wcout << "Loggin request from client: " << c_info.get_client_id() << std::endl;

	// Prepare the packet to inform other client about new client
	ServerClientPacket client_list_packet;
	client_list_packet.set_content_type(packet_content_type::NEW_CONNECTED_CLIENTS);
	client_list_packet.add_to_client_list(c_info);

	WebSocketMessage out_msg(to_utf8string(client_list_packet.as_json().serialize()));

	// Take the lock
	std::unique_lock<std::mutex> lock(m_client_info_mutex);

	m_client_list[c_info.get_client_id()] = c_info;
	m_client_websocket_map[c_info.get_client_id()] = websocket;

	// Send client list to the new client
	send_client_list(c_info.get_client_id());	

	for (auto iter = m_client_list.begin(); iter != m_client_list.end(); iter++)
	{
		if (iter->first == c_info.get_client_id())
		{
			continue;
		}

		std::wcout << "Sending new client login information to client: " << iter->first << std::endl;		
		send_message(iter->first, out_msg);
	}	
}

void ServerManager::handle_client_logout(const ServerClientPacket& packet)
{	
	const ClientInfo& c_info = packet.get_local_client();

	std::wcout << "Logout request from client: " << c_info.get_client_id() << std::endl;

	ServerClientPacket client_list_packet;
	client_list_packet.set_content_type(packet_content_type::NEW_DISCONNECTED_CLIENTS);
	client_list_packet.add_to_client_list(c_info);

	WebSocketMessage out_msg(to_utf8string(client_list_packet.as_json().serialize()));

	// Take the lock
	std::unique_lock<std::mutex> lock(m_client_info_mutex);

	m_client_list.erase(c_info.get_client_id());
	m_client_websocket_map.erase(c_info.get_client_id());

	for (auto iter = m_client_list.begin(); iter != m_client_list.end(); iter++)
	{
		std::wcout << "Sending new client login information to client: " << iter->first << std::endl;
		send_message(iter->first, out_msg);
	}
}

void ServerManager::handle_chat_text_message(const ServerClientPacket& packet)
{
	std::wcout << "Received chat message from client: " << packet.get_sender_client_id() << std::endl;

	const string_t& reveiver_client_id = packet.get_receiver_client_id();
	std::wcout << "Forwarding Chat message to client: " << reveiver_client_id << std::endl;

	WebSocketMessage out_msg(to_utf8string(packet.as_json().serialize()));

	// Take the lock
	std::unique_lock<std::mutex> lock(m_client_info_mutex);
	send_message(reveiver_client_id, out_msg);
}

void ServerManager::send_client_list(const string_t& client_id)
{
	if (m_client_list.size() < 2)
	{
		// Means the client_id is the only client logged on to server.
		return;
	}
	
	std::wcout << "Preparing client-list packet for client: " << client_id << std::endl;

	ServerClientPacket client_list_packet;
	client_list_packet.set_content_type(packet_content_type::NEW_CONNECTED_CLIENTS);

	for (auto iter = m_client_list.begin(); iter != m_client_list.end(); iter++)
	{
		if (iter->first == client_id)
		{
			continue;
		}

		client_list_packet.add_to_client_list(iter->second);
	}

	WebSocketMessage out_msg(to_utf8string(client_list_packet.as_json().serialize()));
	send_message(client_id, out_msg);

	std::wcout << "Client-list sent to client: " << client_id << std::endl;
}

#pragma endregion

#pragma region WebsocketRequestHandler

WebSocketRequestHandler::WebSocketRequestHandler(std::shared_ptr<ServerManager> test_srv)
{
	m_testserver = test_srv;
}

void WebSocketRequestHandler::handleRequest(HTTPServerRequest& request, HTTPServerResponse& response)
{
	try
	{
		std::shared_ptr<WebSocket> websocket_ptr = std::make_shared<WebSocket>(request, response);

		unsigned char buffer[4096];
		int flags, bytes_received;

		do
		{
			Poco::Timespan timeout(/*long seconds*/ 3600, /*long microseconds*/0);
			websocket_ptr->setReceiveTimeout(timeout);
			// If the frame's payload is larger than the provided buffer, a WebSocketException
			//  is thrown and the WebSocket connection must be terminated.
			bytes_received = websocket_ptr->receiveFrame(buffer, sizeof(buffer), flags);

			if ((flags & WebSocket::FRAME_OP_BITMASK) != WebSocket::FRAME_OP_CLOSE)
			{
				WebSocketMessage in_msg(buffer, flags, bytes_received);
				m_testserver->receive_message(websocket_ptr, in_msg);
			}
			else
			{
				std::cout << "Client requested to close the connection" << std::endl;
			}
		}
		while (((flags & WebSocket::FRAME_OP_BITMASK) != WebSocket::FRAME_OP_CLOSE) && bytes_received > 0);
	}
	catch (const WebSocketException& exc)
	{
		// For any errors with HTTP connect, respond to the request. 
		// For any websocket error, exiting this function will destroy the socket.
		switch (exc.code())
		{
		case WebSocket::WS_ERR_HANDSHAKE_UNSUPPORTED_VERSION:
			response.set("Sec-WebSocket-Version", WebSocket::WEBSOCKET_VERSION);
			// fallthrough
		case WebSocket::WS_ERR_NO_HANDSHAKE:
		case WebSocket::WS_ERR_HANDSHAKE_NO_VERSION:
		case WebSocket::WS_ERR_HANDSHAKE_NO_KEY:
			response.setStatusAndReason(HTTPResponse::HTTP_BAD_REQUEST);
			response.setContentLength(0);
			response.send();
			break;
		}
	}
}

#pragma endregion

#pragma region RequestHandlerFactory

RequestHandlerFactory::RequestHandlerFactory(std::shared_ptr<ServerManager> test_srv)
{
	m_testserver = test_srv;
}

HTTPRequestHandler* RequestHandlerFactory::createRequestHandler(const HTTPServerRequest& request)
{
	return new WebSocketRequestHandler(m_testserver);
}

#pragma endregion
