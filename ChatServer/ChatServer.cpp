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
* ChatServer.cpp : Defines the entry point for the console application.
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#include "stdafx.h"
#include "ServerManager.h"

#include "Poco/Net/DNS.h"

using namespace Poco::Net;
using Poco::Net::DNS;
using Poco::Net::HostEntry;

int _tmain(int argc, _TCHAR* argv[])
{
	IPAddress ipv4_address;

	try {
		const HostEntry& entry = DNS::thisHost();
		std::cout << "Canonical Name: " << entry.name() << std::endl;

		for (const auto& alias : entry.aliases())
			std::cout << "Alias: " << alias << std::endl;

		for (const auto& address : entry.addresses()) {
			const bool is_ipv4 = address.family() == IPAddress::IPv4;
			const char* ipv_cstr = (is_ipv4) ? "IPv4" : "IPv6";
			std::cout << "Address (" << ipv_cstr << "): " << address.toString() << std::endl;
			if (is_ipv4) ipv4_address = address;
		}
	}
	catch (...) {
		std::cout << "Host information not available, falling back on 127.0.0.1" << std::endl;
		ipv4_address = IPAddress("127.0.0.1");
	}

	ServerManager server(ipv4_address, 81);

	std::cout << "Press any key to shutdown the server..." << std::endl;
	int x = 0;
	while (x != -1) {
		std::cin >> x;
		if (x >= 0 && x < 255)
			server.broadcast_pose((uint8_t)x);
	}
	return 0;
}

