#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <set>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

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

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
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
            VkExtent2D actualExtent = {WIDTH, HEIGHT};
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

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile(CMAKE_SOURCE_DIR"/shaders/vert.spv");
        auto fragShaderCode = readFile(CMAKE_SOURCE_DIR"/shaders/frag.spv");

        VkShaderModule vertShaderModule;
        VkShaderModule fragShaderModule;

        vertShaderModule = createShaderModule(vertShaderCode);
        fragShaderModule = createShaderModule(fragShaderCode);

        // 显式清除创建的着色器模块对象
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);

        //TODO: create shaderStage
    }

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

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        //显式清除图像视图
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device,swapChain,nullptr);

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

//    std::cout << getenv("VK_ICD_FILENAMES") << std::endl;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
