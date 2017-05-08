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
* ClientInfo.h
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#pragma once

#include "cpprest\json.h"

class ClientInfo
{
public:
	ClientInfo()
	{
	}

	ClientInfo(const utility::string_t& client_id, const utility::string_t& screen_name)
		: m_client_id(client_id), m_screen_name(screen_name)
	{
	}

	ClientInfo(const web::json::value& json_val) :
		m_client_id(json_val.at(U("client_id")).as_string()),
		m_screen_name(json_val.at(U("screen_name")).as_string())
	{
	}

	ClientInfo(const ClientInfo &src) : m_client_id(src.m_client_id), m_screen_name(src.m_screen_name)
	{}

	const utility::string_t& get_client_id() const
	{
		return m_client_id;
	}

	const utility::string_t& get_screen_name() const
	{
		return m_screen_name;
	}

	void set_screen_name(const utility::string_t& screen_name)
	{
		m_screen_name = screen_name;
	}

	web::json::value as_json() const
	{
		web::json::value obj;

		obj[U("client_id")] = web::json::value::string(m_client_id);
		obj[U("screen_name")] = web::json::value::string(m_screen_name);

		return obj;
	}

private:
	utility::string_t  m_client_id;
	utility::string_t  m_screen_name;
};

