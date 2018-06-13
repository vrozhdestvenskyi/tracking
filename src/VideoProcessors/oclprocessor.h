#ifndef OCLPROCESSOR_H
#define OCLPROCESSOR_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

class OclProcessor
{
public:
    OclProcessor();
    ~OclProcessor();

protected:
    cl_int getPlatformId(cl_platform_id &platformId) const;
    cl_int getDeviceId(
        const cl_platform_id &platformId,
        int deviceType,
        cl_device_id &deviceId) const;

    cl_context oclContext_ = NULL;
    cl_command_queue oclQueue_ = NULL;
};

#endif // OCLPROCESSOR_H
