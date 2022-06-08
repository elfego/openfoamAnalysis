#include <string>
#include <cmath>
#include <fstream>

std::vector<T> parse_internal_field_content(content)
{
  bool is_binary = is_binary_format(content);
  for (int ln = 0; ln < content.size(); ++ln) {
    if (content[ln] == "internalField") {

      if (nonuniform) {
        return parse_data_nonuniform(content, ln, len(content), is_binary);
      } else {
        return parse_data_uniform(content[ln]);
      }
    }
  }
}

bool is_binary_format(content, int maxline=20)
{
  for (const std::string lc: content) {
    if ("format" == lc) {
      if ("binary" == lc) {
        return true;
      }
      return false;
    }
  }
  return false;
}
