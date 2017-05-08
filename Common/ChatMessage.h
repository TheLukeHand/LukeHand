#pragma once

#include "cpprest\json.h"

class TextMessage 
{
public:
	TextMessage()
	{}

	TextMessage(const utility::string_t& message) : m_message(message)
	{};

	TextMessage(const web::json::value& jason_val)
	{
		m_message = jason_val[U("text_message")].as_string();
	};

	virtual web::json::value as_json() const
	{
		web::json::value obj;		
		obj[U("text_message")] = web::json::value::string(m_message);
		return obj;
	}

	const utility::string_t& get_message() const 
	{
		return m_message;
	}

	friend std::wostream & operator<< (std::wostream &os, const TextMessage &m);

private:
	utility::string_t m_message;
};

class ChatMessage 
{
public:
	ChatMessage(const utility::string_t& sender_id, const utility::string_t& receiver_id, const TextMessage& message)
		: m_sender_id(sender_id), m_receiver_id(receiver_id), m_message(message) 
	{
		m_timestamp = utility::datetime::utc_now();
	};

	ChatMessage(const web::json::value& jason_val)
	{
		m_sender_id = jason_val[U("sender_id")].as_string();
		m_receiver_id = jason_val[U("receiver_id")].as_string();
		m_timestamp = convutils::datetimeFromJson(jason_val[U("timestamp")]);
		m_message = jason_val[U("message")].as_string();
	};

	utility::string_t get_sender_id() const
	{
		return m_sender_id;
	}
	
	utility::string_t get_receiver_id() const
	{
		return m_receiver_id;
	}

	const TextMessage& get_message() const 
	{
		return m_message;
	}	

	virtual web::json::value as_json() const 
	{
		web::json::value obj;
		obj[U("sender_id")] = web::json::value::string(m_sender_id);
		obj[U("receiver_id")] = web::json::value::string(m_receiver_id);
		obj[U("message")] = m_message.as_json();
		obj[U("timestamp")] = convutils::datetimeToJson(m_timestamp);

		return obj;
	}

	friend std::wostream& operator<< (std::wostream &os, const ChatMessage &m);
	
protected:
	utility::string_t m_sender_id;
	utility::string_t m_receiver_id;	
	utility::datetime m_timestamp;
	TextMessage m_message;
};