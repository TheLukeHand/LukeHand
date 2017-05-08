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
* ProjectUtilities.h
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#pragma once

#include <string.h>

#include "cpprest\json.h"
#include "cpprest\asyncrt_utils.h"

// Default chat server location
#define CHAT_SERVER_URL				U("ws://localhost:9980/")

class ProjectUtilities 
{
public:
	static utility::string_t generate_guid();
	static utility::string_t default_screen_name();
	static void ShowMessage(Platform::String^ message);
	
	static pplx::task<void> async_do_while(std::function<pplx::task<bool>(void)> func);

private:
	static pplx::task<bool> _do_while_iteration(std::function<pplx::task<bool>(void)> func);
	static pplx::task<bool> _do_while_impl(std::function<pplx::task<bool>(void)> func);
};
