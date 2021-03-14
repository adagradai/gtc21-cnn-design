#include "config_parser.h"


bool ConfigParser::isValidConfig(){
	bool config_parsed= false;
	SPDLOG_INFO("model config path {}", parsedConfig.model_config_path);
	SPDLOG_INFO("muxer width {}", parsedConfig.muxer_width);
	SPDLOG_INFO("muxer height {}", parsedConfig.muxer_height);
	SPDLOG_INFO("camera uri {}", parsedConfig.camera_uri);	
	// SPDLOG_INFO("tracker_config_path {}", parsedConfig.tracker_config_path);
	// SPDLOG_INFO("tracker lib path {}", parsedConfig.tracker_lib_path);	

	if(parsedConfig.model_config_path.empty() || !std::experimental::filesystem::exists(parsedConfig.model_config_path))
		SPDLOG_INFO("model config path is empty in json or doesnt exists");				
	else if(parsedConfig.muxer_width <=0 )
		SPDLOG_INFO("invalid muxer width specified in json config");
	else if(parsedConfig.muxer_height <=0)
		SPDLOG_INFO("invalid muxer height specified in json config");	
	// else if(parsedConfig.tracker_config_path.empty() || !std::experimental::filesystem::exists(parsedConfig.tracker_config_path))
	// 	SPDLOG_INFO("tracker config path is empty in json or doesnt exists");			
	// else if(parsedConfig.tracker_lib_path.empty() || !std::experimental::filesystem::exists(parsedConfig.tracker_lib_path))
	// 	SPDLOG_INFO("tracker lib path is empty in json or doesnt exists");				
	else{			
			SPDLOG_INFO("config parsing successful");			
			config_parsed = true;
	}
	return config_parsed;
}


//parse config
void ConfigParser::parseConfig(){
	if(!std::experimental::filesystem::exists(config_path)){
		SPDLOG_INFO("json config doesnt exists!! skipping parsing.");
		return;
	}
	std::ifstream data(config_path);
    data >> jsonConfig;
	
	if(jsonConfig["model_config_path"] != nullptr)
		parsedConfig.model_config_path = jsonConfig["model_config_path"].get<std::string>();
	
	if(jsonConfig["muxer_width"] != nullptr)
		parsedConfig.muxer_width = jsonConfig["muxer_width"].get<int>();

	if(jsonConfig["muxer_height"] != nullptr)
		parsedConfig.muxer_height = jsonConfig["muxer_height"].get<int>();

	if(jsonConfig["camera_uri"] != nullptr)
		parsedConfig.camera_uri = jsonConfig["camera_uri"].get<std::string>();
		
	if(jsonConfig["tracker_config_path"] != nullptr)
		parsedConfig.tracker_config_path = jsonConfig["tracker_config_path"].get<std::string>();

	if(jsonConfig["tracker_lib_path"] != nullptr)
		parsedConfig.tracker_lib_path = jsonConfig["tracker_lib_path"].get<std::string>();	

	if(jsonConfig["stream_output"] != nullptr)
		parsedConfig.stream_output = jsonConfig["stream_output"].get<bool>();
	else
		//disable streaming by default
		parsedConfig.stream_output= false;
}
