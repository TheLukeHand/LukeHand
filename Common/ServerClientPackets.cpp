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
* ServerClientPackets.cpp
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#include "ServerClientPackets.h"

using namespace web;

ServerClientPacket::ServerClientPacket() 
{
	m_content_type = NONE;
}

ServerClientPacket::ServerClientPacket(const json::value& json_val)
{
	m_content_type = static_cast<packet_content_type>( json_val.at(U("content_type")).as_integer());

	if (m_content_type == CHAT_TEXT_MSG)
	{
		m_sender_client = json_val.at(U("sender_client")).as_string();
		m_receiver_client = json_val.at(U("receiver_client")).as_string();
		m_chat_text = json_val.at(U("chat_text")).as_string();
	}
	else if (m_content_type == CLIENT_LOGIN || m_content_type == CLIENT_LOGOUT)
	{
		m_local_client = ClientInfo(json_val.at(U("local_client")));
	}
	else if (m_content_type == NEW_CONNECTED_CLIENTS || m_content_type == NEW_DISCONNECTED_CLIENTS)
	{
		json::array client_list(json_val.at(U("client_list")).as_array());

		for (auto iter = client_list.cbegin(); iter != client_list.cend(); ++iter)
		{
			ClientInfo c_info(*iter);
			m_client_list.push_back(c_info);
		}
	}
}

web::json::value ServerClientPacket::as_json() const
{
	web::json::value obj;
	obj[U("content_type")] = json::value::number(m_content_type);

	if (m_content_type == CHAT_TEXT_MSG)
	{
		obj[U("sender_client")] = web::json::value::string(m_sender_client);
		obj[U("receiver_client")] = web::json::value::string(m_receiver_client);
		obj[U("chat_text")] = web::json::value::string(m_chat_text);
	}
	else if (m_content_type == CLIENT_LOGIN || m_content_type == CLIENT_LOGOUT)
	{
		obj[U("local_client")] = m_local_client.as_json();
	}
	else if (m_content_type == NEW_CONNECTED_CLIENTS || m_content_type == NEW_DISCONNECTED_CLIENTS)
	{
		json::value client_list = web::json::value::array(m_client_list.size());

		for (unsigned int i = 0; i < m_client_list.size(); i++)
		{
			client_list[i] = m_client_list[i].as_json();
		}

		obj[U("client_list")] = client_list;
	}

	return obj;
}

void ServerClientPacket::set_content_type(packet_content_type content_type)
{
	m_content_type = content_type;
}

packet_content_type ServerClientPacket::get_content_type()
{
	return m_content_type;
}

const utility::string_t& ServerClientPacket::get_sender_client_id() const 
{
	return m_sender_client;
}

const utility::string_t& ServerClientPacket::get_receiver_client_id() const
{
	return m_receiver_client;
}

void ServerClientPacket::set_sender_client_id(const utility::string_t& client_id)
{
	m_sender_client = client_id;
}

void ServerClientPacket::set_receiver_client_id(const utility::string_t& client_id)
{
	m_receiver_client = client_id;
}

void ServerClientPacket::add_to_client_list(const ClientInfo& client_info)
{
	m_client_list.push_back(client_info);
}

const std::vector<ClientInfo>& ServerClientPacket::get_client_list() const 
{
	return m_client_list;
}

void ServerClientPacket::set_chat_text(const utility::string_t& chat_text)
{
	m_chat_text = chat_text;
}

const utility::string_t& ServerClientPacket::get_chat_text() const 
{
	return m_chat_text;
}

void ServerClientPacket::set_local_client(const ClientInfo& local_client)
{
	m_local_client = local_client;
}

const ClientInfo& ServerClientPacket::get_local_client() const
{
	return m_local_client;
}