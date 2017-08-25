#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

UMat loadRaw(const string &path, const uint height, const uint width) {
  vector<char> model;
  ifstream file(path, ios_base::binary);

  file.seekg(0, std::ios::end);
  std::streampos length(file.tellg());
  file.seekg(0, std::ios::beg);

  model.resize(static_cast<std::size_t>(length));
  file.read(reinterpret_cast<char *>(&model.front()),
            static_cast<std::size_t>(length));

  UMat uraw;
  const auto raw = Mat(height, width, CV_8UC1, model.data());
  raw.copyTo(uraw);

  return uraw;
}

int main(int argc, char **argv) {
  auto keys = "{help h usage ? | | Display this message}"
              "{input i | | Input Raw Bayer image}"
              "{output o | | Output GrayScale image (optional)}"
              "{height | | Image height (default 3648)}"
              "{width | | Image width (default 4912)}"
              "{nocl | | Set to not use OpenCL (optional)}"
              "{waitkey wk | | Wait key timer. If not set, no image will be "
              "displayed (optional).}";
  auto cmd = CommandLineParser(argc, argv, keys);

  if (cmd.has("help")) {
    cmd.printMessage();
    return 0;
  }

  if (!cmd.has("nocl"))
    ocl::setUseOpenCL(true);

  uint width = 4912, height = 3684;
  if (cmd.has("height"))
    height = cmd.get<uint>("height");

  if (cmd.has("width"))
    width = cmd.get<uint>("width");

  UMat raw, gray;
  if (cmd.has("input")) {
    raw = loadRaw(cmd.get<string>("input"), height, width);
  } else {
    cerr << "You must provide an input image." << endl;
    return -1;
  }

  cvtColor(raw, gray, CV_BayerGR2GRAY);

  if (cmd.has("waitkey")) {
    imshow("Gray", gray);
    waitKey(cmd.get<int>("waitkey"));
  }

  if (cmd.has("output"))
    imwrite(cmd.get<string>("output"), gray);

  return 0;
}
