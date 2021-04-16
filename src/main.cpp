#include "config_parser.h"
#include "face_blur_pipeline.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#define CONFIG_PATH "../config/init_config.json"
#define LOGNAME_FORMAT "logs/ds_face_blur_sample_%Y%m%d_%H%M%S.log"
#define LOGNAME_SIZE 40

int main(){
	ConfigParser parser(CONFIG_PATH);
	parser.parseConfig();
	if(!parser.isValidConfig())
		return -1;
	
	static char name[LOGNAME_SIZE];
    time_t      now = time(0);
    strftime(name, sizeof(name), LOGNAME_FORMAT, localtime(&now));
    auto file_logger = spdlog::basic_logger_mt("DS face blur sample version 0.24", name);
    file_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%-5!l] [%-20!s] [%-21!!] %v");
    file_logger->flush_on(spdlog::level::info);
    spdlog::set_default_logger(file_logger);
	
	gst_init(nullptr, nullptr);
	FaceBlurPipeline face_blur_pipeline(parser.getConfig());
	if(!face_blur_pipeline.createElements()){
		SPDLOG_INFO("failed to create elements");
		return -1;
	}
	if(!face_blur_pipeline.setElementProperties())
	{
		SPDLOG_INFO("failed to set element properties!!");
		return -1;
	}
	if(!face_blur_pipeline.linkElements()){
		SPDLOG_INFO("failed to link elements");
		return -1;
	}
	face_blur_pipeline.start();
	face_blur_pipeline.stop();
	return 0;
}