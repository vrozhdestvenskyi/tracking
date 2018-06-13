#include <oclprocessor.h>
#include <array>
#include <QDebug>

OclProcessor::OclProcessor()
{
    cl_platform_id platformId;
    if (getPlatformId(platformId) != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to initialize cl_platform_id");
    }
    cl_device_id deviceId;
    if (getDeviceId(platformId, CL_DEVICE_TYPE_GPU, deviceId) != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to initialize cl_device_id");
    }
    oclContext_ = clCreateContext(NULL, 1, &deviceId, NULL, NULL, NULL);
    if (!oclContext_)
    {
        throw std::runtime_error("Failed to initialize cl_context");
    }
    oclQueue_ = clCreateCommandQueue(oclContext_, deviceId,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);
    if (!oclQueue_)
    {
        throw std::runtime_error("Failed to initialize cl_command_queue");
    }
}

OclProcessor::~OclProcessor()
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
    if (clGetPlatformIDs(ids.size(), ids.data(), &idsCount) != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to get platform ids");
    }
    for (cl_uint i = 0; i < idsCount; ++i)
    {
        std::array<cl_char, 256> info;
        size_t infoLength = 0;
        if (clGetPlatformInfo(ids[i], CL_PLATFORM_VERSION, info.size(),
                (void*)info.data(), &infoLength) != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to get platform info");
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
    if (clGetDeviceIDs(platformId, deviceType, ids.size(), ids.data(), &idsCount) != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to get device ids");
    }
    if (idsCount > 0)
    {
        deviceId = ids[0];
        std::array<cl_char, 256> info;
        size_t infoLength = 0;
        if (clGetDeviceInfo(deviceId, CL_DEVICE_NAME, info.size(),
                (void*)info.data(), &infoLength) != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to get device info");
        }
        qDebug("Use computation device: %s", info.data());
        return CL_SUCCESS;
    }
    return CL_INVALID_DEVICE;
}

