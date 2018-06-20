#include <oclprocessor.h>
#include <array>
#include <QDebug>

OclProcessor::~OclProcessor()
{
    release();
}

cl_int OclProcessor::initialize()
{
    cl_platform_id platformId;
    cl_device_id deviceId;
    if (getPlatformId(platformId) == CL_SUCCESS &&
        getDeviceId(platformId, CL_DEVICE_TYPE_GPU, deviceId) == CL_SUCCESS)
    {
        oclContext_ = clCreateContext(NULL, 1, &deviceId, NULL, NULL, NULL);
    }
    if (oclContext_)
    {
        oclQueue_ = clCreateCommandQueue(oclContext_, deviceId,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);
    }
    if (!oclQueue_)
    {
        release();
        return CL_INVALID_COMMAND_QUEUE;
    }
    return CL_SUCCESS;
}

void OclProcessor::release()
{
    if (oclQueue_)
    {
        clReleaseCommandQueue(oclQueue_);
        oclQueue_ = NULL;
    }
    if (oclContext_)
    {
        clReleaseContext(oclContext_);
        oclContext_ = NULL;
    }
}

cl_int OclProcessor::getPlatformId(cl_platform_id &platformId) const
{
    cl_uint idsCount = 0;
    std::array<cl_platform_id, 4> ids;
    cl_int status = clGetPlatformIDs(ids.size(), ids.data(), &idsCount);
    if (status != CL_SUCCESS)
    {
        return status;
    }
    for (cl_uint i = 0; i < idsCount; ++i)
    {
        std::array<cl_char, 256> info;
        size_t infoLength = 0;
        status = clGetPlatformInfo(ids[i], CL_PLATFORM_VERSION, info.size(),
            (void*)info.data(), &infoLength);
        if (status != CL_SUCCESS)
        {
            return status;
        }
        if (infoLength == 12 && std::string(info.data(), info.data() + 11) == "OpenCL 1.2 ")
        {
            qDebug("OpenCL platform version: %s", info.data());
            platformId = ids[i];
            return CL_SUCCESS;
        }
    }
    return CL_INVALID_PLATFORM;
}

cl_int OclProcessor::getDeviceId(
    const cl_platform_id &platformId,
    int deviceType,
    cl_device_id &deviceId) const
{
    cl_uint idsCount = 0;
    std::array<cl_device_id, 8> ids;
    cl_int status = clGetDeviceIDs(platformId, deviceType, ids.size(), ids.data(), &idsCount);
    if (status != CL_SUCCESS)
    {
        return status;
    }
    if (idsCount > 0)
    {
        deviceId = ids[0];
        std::array<cl_char, 256> info;
        size_t infoLength = 0;
        status = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, info.size(),
            (void*)info.data(), &infoLength);
        if (status != CL_SUCCESS)
        {
            return status;
        }
        qDebug("Use computation device: %s", info.data());
        return CL_SUCCESS;
    }
    return CL_INVALID_DEVICE;
}

