add_library(
	deltagpu SHARED
	....cu
)

target_compile_options(deltagpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --debug --device-debug -Xcompiler -lineinfo -rdynamic>)
target_include_directories(deltagpu PUBLIC "../include")
