#include <oclprocessor.h>
#include <array>
#include <QDebug>
#include <QFile>

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
    if (oclQueue_)
    {
        std::string kernelSourceStr = std::move(getKernelSource());
        const char *kernelSource[] = { kernelSourceStr.c_str() };
        oclProgram_ = clCreateProgramWithSource(oclContext_, 1, kernelSource, NULL, NULL);
    }
    // TODO: investigate different optimization flags from khronos-opencl-1.1
    // at section 5.6.3
    if (!oclProgram_)
    {
        release();
        return CL_INVALID_PROGRAM;
    }
    if (clBuildProgram(oclProgram_, 1, &deviceId, "-cl-std=CL1.2", NULL, NULL) != CL_SUCCESS)
    {
        const size_t logSizeMax = 32 * 1024;
        char log[logSizeMax];
        size_t logSize = 0;
        if (clGetProgramBuildInfo(oclProgram_, deviceId, CL_PROGRAM_BUILD_LOG, logSizeMax,
                log, &logSize) == CL_SUCCESS)
        {
            qDebug("\nOpenCL build log:\n%s\n", log);
        }
        else
        {
            qDebug("clGetProgamBuildInfo(...) failed");
        }
        release();
        return CL_BUILD_PROGRAM_FAILURE;
    }
    return CL_SUCCESS;
}

void OclProcessor::release()
{
    if (oclProgram_)
    {
        clReleaseProgram(oclProgram_);
        oclProgram_ = NULL;
    }
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

std::string OclProcessor::getKernelSource() const
{
    std::string kernelSource;
    for (const std::string &path : kernelPaths_)
    {
        QFile f(path.c_str());
        if (!f.open(QFile::ReadOnly))
        {
            return "Could not open: " + path + " for reading\n";
        }
        kernelSource += f.readAll().toStdString() + "\n\n\n";
    }
    return kernelSource;
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
        if (infoLength == 12 && std::string(info.data(), info.data() + 11) == "OpenCL 2.1 ")
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

