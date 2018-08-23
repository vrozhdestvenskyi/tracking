#ifndef OCLPROCESSOR_H
#define OCLPROCESSOR_H

#include <vector>
#include <string>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

class OclProcessor
{
public:
    ~OclProcessor();

protected:
    cl_int initialize();
    void release();

    std::string getKernelSource() const;

    cl_int getPlatformId(cl_platform_id &platformId) const;
    cl_int getDeviceId(
        const cl_platform_id &platformId,
        int deviceType,
        cl_device_id &deviceId) const;

    std::vector<std::string> kernelPaths_;
    cl_context oclContext_ = NULL;
    cl_command_queue oclQueue_ = NULL;
    cl_program oclProgram_ = NULL;
};

#endif // OCLPROCESSOR_H
