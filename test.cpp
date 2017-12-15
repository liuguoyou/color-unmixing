#include <string>
#include "unmixing/unmixing.h"

int main(int argc, char** argv)
{
    const std::string image_file_path(argv[1]);
    const std::string output_directory_path(argv[2]);

    ColorUnmixing::compute_color_unmixing(image_file_path, output_directory_path);

    return 0;
}
