#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <set>

//顶点数据结构体
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
};

//队列族返回值结构体
struct QueueFamilyIndices {
    int graphicsFamily = -1;
    int presentFamily = -1;

    bool isComplete() {
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
};

//交换链信息结构体
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;//基础表面特性
    std::vector<VkSurfaceFormatKHR> formats;//表面格式
    std::vector<VkPresentModeKHR> presentModes;//可用的呈现模式

};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

//可并行处理的最大帧数
const int MAX_FRAME_IN_FLIGHT = 2;

//校验层扩展
const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
};

//交换链扩展
#ifdef APPLE
const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset"
};
#else
const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#endif

#ifndef NDEBUG //Mac默认没有DEBUG这个宏，所以都是假的
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

const std::vector<Vertex> vertices = {
        {{-0.5f,-0.5f},{1.0f,0.0f,0.0f}},
        {{0.5f, -0.5f},{0.0f,1.0f,0.0f}},
        {{0.5f,0.5f},{0.0f,0.0f,1.0f}},
        {{-0.5f,0.5f},{1.0f,1.0f,1.0f}}
};

const std::vector<uint16_t> indices = {
        0,1,2,2,3,0
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkDevice device;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    size_t currentFrame = 0;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    bool framebufferResized = false;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffer();
        createCommandBuffers();
        createSyncObjects();
    }

    //创建窗口表面
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    //找物理设备
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;

        //查询显卡设备
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        //如果找不到能用的显卡
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);

        //存储设备对象
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        //找符合需求的显卡
        for (const auto &device: devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    //创建逻辑设备
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }



        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
    }

    //检查设备是否符合要求
    bool isDeviceSuitable(VkPhysicalDevice device) {

        /*
         * VkPhysicalDeviceProperties deviceProperties;基本的设备属性
         * vkGetPhysicalDeviceProperties(device,&deviceProperties);
         * VkPhysicalDeviceFeatures deviceFeatures;设备对于特性的支持
         * vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
         * 例：设备要求满足几何着色器(还真不支持)。此外可以按照设备的特性加权取高分选用设备或由用户自行选择
         * return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
         */

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        //确保设备支持需要指令的队列族
        QueueFamilyIndices indices = findQueueFamilies(device);

        //设备对扩展的支持
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        //注意只能在验证交换链扩展可用后查询交换链的细节信息
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            //交换链支持至少一种图像格式和一种窗口表面呈现模式即可
            swapChainAdequate = !swapChainSupport.formats.empty()
                                && !swapChainSupport.presentModes.empty();
        }
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    //检查设备对扩展的支持
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        //所需扩展保存在集合中
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        //枚举可用扩展，若集合元素为0则全部满足
        for (const auto &extension: availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
        //设备若支持呈现队列，就一定支持交换链，但显式检测并启用交换链较优
    }

    //寻找符合要求的队列族
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto &queueFamily: queueFamilies) {

            //找到一个支持呈现队列的队列族
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    //查询交换链支持细节
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        //查询基础表面特性
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);//查询表面特性

        //查询表面支持格式
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }

    //创建交换链
    void createSwapChain() {
        SwapChainSupportDetails swapchainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

        //使用交换链支持的最小图像个数+1 数量的图像来实现三倍缓冲
        uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
        // maxImageCount == 0 表明只要内存可以满足，可以使用任意数量图像
        if (swapchainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapchainSupport.capabilities.maxImageCount) {
            imageCount = swapchainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        //图像包含的层次，VR应用中会使用更多层次
        createInfo.imageArrayLayers = 1;
        //在图像中会进行的操作，此处为颜色附着,如果将图像用于后处理，可使用VK_IMAGE_USAGE_TRANSFER_DST_BIT
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {(uint32_t) indices.graphicsFamily, (uint32_t) indices.presentFamily};

        /*
         *  CONCURRENT 一张图像可以在多个队列族中使用
         *  EXCLUSIVE   一张图像只能在一个队列族中使用，除非显式改变图像所有权
         */

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else{
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        //指定固定变换操作，如旋转90度
        createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
        //alpha通道是否用于混合,此处忽略alpha通道
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        //呈现模式
        createInfo.presentMode = presentMode;
        //是否不需要计算被窗口系统中其他窗口遮挡的像素颜色，若如此设置，回读窗口像素值可能出现问题
        createInfo.clipped = VK_TRUE;
        //重建交换链时需要之前的交换链
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    //重建交换链
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }

    //选择表面格式
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
        /*
         * VkSurfaceFormatKHR 条目包含format和colorSpace
         * format指定颜色通道和存储类型
         * 例VK_FORMAT_B8G8R8A8_UNORM表示通道以BGRA顺序，8位无符号整数
         * colorSpace表示SRGB是否被支持，SRGB可以得到更准确的颜色表示
         */

        //如果无首选格式
        if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        }

        //返回格式列表则检查
        for (const auto &availableFormat: availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        //不能找到格式则 1.根据加权选择格式 2. 直接使用列表中的第一个格式
        return availableFormats[0];
    }

    //查找最佳的可用呈现模式
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {

        /*
        * VK_PRESENT_MODE_IMMEDIATE_KHR 应用程序提交的图像直接传输到屏幕上，可能导致撕裂
        * VK_PRESENT_MODE_FIFO_KHR 经典的缓冲队列
        * VK_PRESENT_MODE_FIFO_RELAXED_KHR 与缓冲队列的区别是，如应用延迟导致垂直回扫时队列为空，在下一次回扫前的提交将被立即显示
        * VK_PRESENT_MODE_MAILBOX_KHR 不会在队列满时阻塞应用程序，而是直接将队列中的图像替换为新提交的图像，可用于实现三倍缓冲（？）
        * 优先使用MAILBOX，其次IMMEDIATE，其次FIFO（因为部分驱动对FIFO支持的不好）
        */

        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;
        for (const auto &availablePresentMode: availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                bestMode = availablePresentMode;
            }
        }
        return bestMode;
    }

    //选择交换范围，也即图像分辨率
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {

        // 如果窗口系统设置可用分辨率为unit32_t最大值，表示可自由选择最合适的交换范围，只需在maxImageExtent，minImageExtent范围内
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
            actualExtent.width = std::max(capabilities.minImageExtent.width,
                                          std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height,
                                           std::min(capabilities.maxImageExtent.height, actualExtent.height));
            return actualExtent;
        }
    }

    //为交换链中的图像建立图像视图
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            //viewType表示图像被视作一维、二维、三维纹理或立方体贴图
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // 图像颜色通道映射
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            //指定图像用途和图像可被访问部分
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    //读取文件
    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(),fileSize);

        file.close();

        return buffer;
    }

    //渲染管线
    void createGraphicsPipeline() {

        //着色器

        //读取着色器代码
        auto vertShaderCode = readFile(CMAKE_SOURCE_DIR"/shaders/vert.spv");
        auto fragShaderCode = readFile(CMAKE_SOURCE_DIR"/shaders/frag.spv");

        VkShaderModule vertShaderModule;
        VkShaderModule fragShaderModule;

        vertShaderModule = createShaderModule(vertShaderCode);
        fragShaderModule = createShaderModule(fragShaderCode);

        //创建顶点着色器阶段
        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        //指定阶段调用的着色器函数
        vertShaderStageInfo.pName = "main";
        //指定着色器用到的常量
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        //创建片元着色器阶段
        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        //指定阶段调用的着色器函数
        fragShaderStageInfo.pName = "main";
        //指定着色器用到的常量
        fragShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo shaderStages[] = {
                vertShaderStageInfo,
                fragShaderStageInfo
        };


        //顶点输入

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        //描述顶点数据组织信息的结构体数组
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();


        //输入装配

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
        inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        /*
         * POINT_LIST       点图元
         * LINE_LIST        每两个顶点构成一个线段图元
         * LINE_STRIP       首尾相连的线段，除第一个外每个使用上一个线段图元的一个顶点
         * TRIANGLE_LIST    每三个构成一个三角形
         * TRIANGLE_STRIP   每个三角形的第二个和第三个顶点被下一个三角形用于前两个顶点
         */
        inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        /*
         * 通过索引缓冲复用顶点缓冲中的顶点数据，
         * 如设为true且使用_STRIP,可通过特殊索引值重启图元，将之后的索引值设置为新图元的第一个顶点
         *适用于多个短的strip图元类型，少量较长的图元类型应当使用多个绘制命令
         */
        inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

        //视口和裁剪

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        //光栅化

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

        //是否将近远平面外的片段截断在近平面远平面上而非丢弃
        rasterizer.depthClampEnable = VK_FALSE;

        //禁止一切片段输入到帧缓冲
        rasterizer.rasterizerDiscardEnable = VK_FALSE;

        /*
         * VK_POLYGON_MODE_FILL 整个多边形都产生片段
         * VK_POLYGON_MODE_LINE 仅边产生片段
         * VK_POLYGON_MODE_FILL 仅顶点产生片段
         */
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

        rasterizer.lineWidth = 1.0f;
        //表面剔除设置
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        //指定顺时针/逆时针顶点序为正面
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        //设置添加一个常量或基于斜率得到的变量值到深度值上
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        //多重采样

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        //深度和模版测试

        {
            //禁用
        }

        //颜色混合

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        //动态状态

        VkDynamicState dynamicStates[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_LINE_WIDTH
        };

        VkPipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        //管线布局

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        //创建渲染管线

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        //用于以一个创建好的管线为基础创建新的管线
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // 显式清除创建的着色器模块对象

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    //着色器模块
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // 注意需要将存储字节码的数组指针转换为const uint32_t*类型以匹配结构体中的变量类型
        // 且指针指向的地址应符合uint32_t的内存对齐方式，此处使用std::vector
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    //帧缓冲
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                    swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    //渲染流程
    void createRenderPass() {
        //附着
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        /*
         * LOAD         保持附着的现有内容
         * CLEAR        使用常量值清除附着内容
         * DONT_CARE

         * STORE        渲染内容被存储以便之后读取
         * DONT_CARE    渲染后不读取帧缓冲内容
         */

        //对颜色和深度缓冲起效
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        //对模版缓冲起效
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        //渲染流程开始前/结束后的图像布局方式
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        //子流程和附着引用
        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        // 配置子流程依赖
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;

        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        //创建渲染流程
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    //指令缓冲
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
        poolCreateInfo.flags = 0;

        if (vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    //创建指令缓冲
    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());
        VkCommandBufferAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        //记录指令到指令缓冲
        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            beginInfo.pInheritanceInfo = nullptr;

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            //开始渲染流程
            VkRenderPassBeginInfo renderPassBeginInfo = {};
            renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassBeginInfo.renderPass = renderPass;
            renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];

            //渲染区域，通常设置为和附着大小一致
            renderPassBeginInfo.renderArea.offset = {0,0};

            renderPassBeginInfo.renderArea.extent = swapChainExtent;
            VkClearValue clearValue = {0.0f, 0.0f, 0.0f, 1.0f};
            renderPassBeginInfo.clearValueCount = 1;
            renderPassBeginInfo.pClearValues = &clearValue;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

            /*
             * vertexCount
             * instanceCount    为1，不进行实例渲染
             * firstVertex      gl_VertexIndex
             * firstInstance    gl_InstanceIndex
             */
            vkCmdEndRenderPass(commandBuffers[i]);
            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                     stagingBufferMemory);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize =sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,stagingBuffer,stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize,0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer,indexBuffer,bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffer() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    // 选择最合适的内存类型
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags propertyFlags) {
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,properties);

        if (vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);

    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        //开始记录内存传输指令
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        //缓冲复制操作
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        //提交指令缓冲完成传输操作的执行
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    //描述符集布局
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;

        //指定描述符被使用的着色器阶段
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount = 1;
        layoutCreateInfo.pBindings = &uboLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float,std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f,0.0f,1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f,2.0f,2.0f),glm::vec3(0.0f,0.0f,0.0f),glm::vec3(0.0f,0.0f,1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f),swapChainExtent.width/(float)swapChainExtent.height, 0.1f,10.0f);
        //GLM最初为OpenGL设计，裁剪坐标Y轴与vulkan相反
        ubo.proj[1][1] *= -1;

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame],
                        VK_TRUE, std::numeric_limits<uint64_t>::max());
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;

        //从交换链获取一张图像
        VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(),
                              imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(imageIndex);

        //提交指令缓冲
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        //开始时需要等待的信号量和管线阶段
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //指定被提交的指令缓冲对象
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        //执行后发出的信号量对象
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAME_IN_FLIGHT;

    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAME_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAME_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAME_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreCreateInfo = {};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
                || vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
                || vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }

    }

    void cleanupSwapChain() {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for(size_t i = 0; i < swapChainImageViews.size(); i++) {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    //创建Vulkan实例
    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        appInfo.pNext = nullptr;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredCreateInstanceExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
#ifdef APPLE
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    std::vector<const char*> getRequiredCreateInstanceExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        //VK_KHR_surfaceVK_EXT_metal_surface

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
#ifdef APPLE
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

#else
#endif

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
