cmake_minimum_required(VERSION 3.14)
project(DocumentValidator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_PREFIX_PATH 
    "/opt/homebrew/opt/qt@5/lib/cmake"
)

set(CMAKE_MACOSX_BUNDLE ON)
set(MACOSX_BUNDLE_ICON_FILE logo.icns)
set_source_files_properties(gui/resources/icons/logo.icns PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources")

# Qt
find_package(Qt5 REQUIRED COMPONENTS Widgets Multimedia)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
qt5_add_resources(QT_RESOURCES gui/resources/resources.qrc)

# nlohmann/json (header-only библиотека)
find_package(nlohmann_json 3.11.0 QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann/json найдена через find_package")
else()
    # Если не найдена через find_package, попробуем через Homebrew
    if(EXISTS "/opt/homebrew/include/nlohmann/json.hpp")
        include_directories("/opt/homebrew/include")
        message(STATUS "nlohmann/json найдена в /opt/homebrew/include")
    else()
        message(FATAL_ERROR "nlohmann/json не найдена. Установите: brew install nlohmann-json")
    endif()
endif()

# Document Validator library
add_library(DocumentValidatorLib
        src/DocumentValidator.cpp
        include/DocumentValidator.h
)
target_include_directories(DocumentValidatorLib PUBLIC include)

# Линковка nlohmann/json если найдена через find_package
if(nlohmann_json_FOUND)
    target_link_libraries(DocumentValidatorLib PUBLIC nlohmann_json::nlohmann_json)
endif()

# GUI app
add_executable(maboy
        gui/src/main.cpp
        gui/src/MainWindow.cpp
        gui/include/MainWindow.h
        gui/include/FileDropPanel.h
        gui/src/FileDropPanel.cpp
        gui/include/ErrorPopup.h
        gui/src/ErrorPopup.cpp
        gui/include/AnimatedButton.h
        gui/src/AnimatedButton.cpp
        ${QT_RESOURCES}
        gui/resources/icons/logo.icns
        gui/resources/fronts/IBMPlexMono.ttf
)

target_link_libraries(maboy
        Qt5::Widgets
        Qt5::Multimedia
        DocumentValidatorLib
)
