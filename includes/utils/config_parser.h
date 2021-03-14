#ifndef __CONFIG_PARSER_H__
#define __CONFIG_PARSER_H__

#include <string>
#include <nlohmann/json.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

using jsonlib = nlohmann::json;


class FrameCropDims{
	public:
	int x;
	int y;
	int w;
	int h;
};

class JsonConfig{
	public:		
	const uint tracker_height = 480;
	const uint tracker_width = 640;

	std::string model_config_path;
	uint muxer_height;
	uint muxer_width;	
	std::string camera_uri;
	std::string tracker_lib_path;
	std::string tracker_config_path;
	bool stream_output;
};


class ConfigParser{
private:
	JsonConfig parsedConfig;
	jsonlib jsonConfig;
	std::string config_path;
	bool config_parsed;
	
public:
	ConfigParser(std::string init_config_path){
		config_path = init_config_path;
		config_parsed = false;
	}
	bool isValidConfig();
	void parseConfig();
	JsonConfig getConfig() { return parsedConfig;}
};

#endif 