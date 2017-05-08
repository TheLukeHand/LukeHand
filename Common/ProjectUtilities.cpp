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
* ProjectUtilities.cpp
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#include "ProjectUtilities.h"

using namespace Windows::Foundation::Collections;
using namespace Windows::Networking;
using namespace Windows::Networking::Connectivity;
using namespace Platform;

utility::string_t ProjectUtilities::generate_guid()
{
	GUID guid;
	CoCreateGuid(&guid);

	OLECHAR guid_buff[40];
	StringFromGUID2(guid, guid_buff, 40);
	
	utility::string_t guid_str(guid_buff);
	return guid_str;
}

utility::string_t ProjectUtilities::default_screen_name()
{
	IVectorView<HostName^>^ host_name_list = NetworkInformation::GetHostNames();

	String^ screen_name_str_p = "MyScreenName";

	for (unsigned int i = 0; i < host_name_list->Size; i++)
	{
		if (host_name_list->GetAt(i)->Type == HostNameType::DomainName)
		{
			screen_name_str_p = host_name_list->GetAt(i)->DisplayName;
			break;
		}
	}

	utility::string_t screen_name(screen_name_str_p->Data());
	return screen_name.substr(0, screen_name.find_first_of('.'));
}

void ProjectUtilities::ShowMessage(Platform::String^ message)
{
	// Create the message dialog and set its content
	Windows::UI::Popups::MessageDialog^ msgDiag = ref new Windows::UI::Popups::MessageDialog(message);

	// Add OK commands 
	Windows::UI::Popups::UICommand^ continueCommand = ref new Windows::UI::Popups::UICommand("OK");

	// Add the commands to the dialog
	msgDiag->Commands->Append(continueCommand);

	// Show the message dialog
	msgDiag->ShowAsync();
}

pplx::task<void> ProjectUtilities::async_do_while(std::function<pplx::task<bool>(void)> func)
{
	return _do_while_impl(func).then([](bool){});
}

pplx::task<bool> ProjectUtilities::_do_while_iteration(std::function<pplx::task<bool>(void)> func)
{
	pplx::task_completion_event<bool> ev;
	
	func().then([=](bool task_completed)
	{
		ev.set(task_completed);
	});
	
	return pplx::create_task(ev);
}

pplx::task<bool> ProjectUtilities::_do_while_impl(std::function<pplx::task<bool>(void)> func)
{
	return _do_while_iteration(func).then([=](bool continue_next_iteration) -> pplx::task<bool>
	{
		return ((continue_next_iteration == true) ? _do_while_impl(func) : pplx::task_from_result(false));
	});
}
