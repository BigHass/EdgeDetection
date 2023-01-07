#pragma once

#include <vector>
#include <memory>
#include <span>
#include "stb_image.h"

namespace h2o {

    class Image {
        int m_rows; // The number of rows in the image.
        int m_columns; // The number of columns in the image.
        std::vector<std::byte> m_data; // The pixel data of the image.


    public:
        explicit Image(int rows, int columns, std::vector<std::byte> data) :
                m_rows(rows),
                m_columns(columns),
                m_data(std::move(data)) {

        }

        explicit Image(int rows, int columns) :
                m_rows(rows),
                m_columns(columns) {

            m_data.resize(m_rows * m_columns);
        }
        // A shared pointer to an Image.
        using Ptr = std::shared_ptr<Image>;

        static Ptr empty(int rows, int columns) {

            return std::make_shared<Image>(rows, columns);
        }

        static Ptr from_file(std::string_view path) {

            int rows{}, cols{}, channels{};
            auto image = stbi_load(path.data(), &cols, &rows, &channels, 1);
            std::vector<std::byte> data(rows * cols);
            for (size_t i = 0; i < rows * cols; ++i)
                data[i] = std::byte(image[i]);

            stbi_image_free(image);

            return std::make_shared<Image>(
                    rows,
                    cols,
                    std::move(data)
            );
        }

        [[nodiscard]]
        const std::byte &at(size_t index) const {

            return m_data.at(index);
        }

        [[nodiscard]]
        std::byte &at(size_t index) {

            return m_data.at(index);
        }

        [[nodiscard]]
        int rows() const {

            return m_rows;
        }

        [[nodiscard]]
        int columns() const {

            return m_columns;
        }

        [[nodiscard]]
        std::span<const std::byte> span() const {

            return {m_data.data(), m_data.size()};
        }
        // Returns a pointer to the pixel data of the image, allowing it to be modified.
        std::byte *mut_data() {

            return m_data.data();
        }
    };

}
