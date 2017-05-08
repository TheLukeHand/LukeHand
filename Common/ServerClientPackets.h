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
* ServerClientPackets.h
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#pragma once

#include "ClientInfo.h"

// Indicates of messages exchanged between client and server
typedef enum
{
	NONE,
	CLIENT_LOGIN,
	CLIENT_LOGOUT,
	NEW_CONNECTED_CLIENTS,
	NEW_DISCONNECTED_CLIENTS,
	CHAT_TEXT_MSG

} packet_content_type;
	
class ServerClientPacket 
{
public:
	ServerClientPacket();
	ServerClientPacket(const web::json::value& json_val);

	web::json::value as_json() const;

	void set_content_type(packet_content_type content_type);
	packet_content_type get_content_type();

	void set_sender_client_id(const utility::string_t& client_id);
	const utility::string_t& get_sender_client_id() const;

	void set_receiver_client_id(const utility::string_t& client_id);
	const utility::string_t& get_receiver_client_id() const;

	void add_to_client_list(const ClientInfo& client_info);
	const std::vector<ClientInfo>& get_client_list() const;	

	void set_chat_text(const utility::string_t& chat_text);
	const utility::string_t& get_chat_text() const;

	void set_local_client(const ClientInfo& local_client);
	const ClientInfo& get_local_client() const;

private:
	packet_content_type m_content_type;
	utility::string_t m_sender_client;
	utility::string_t m_receiver_client;

	utility::string_t m_chat_text;
	ClientInfo m_local_client;
	std::vector<ClientInfo> m_client_list;
};
