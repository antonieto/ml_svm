library(readr)

# Specify the row to plot
row <- 10

train <- read_csv("../train.csv", show_col_types = FALSE)
train[1, 1]
img <- matrix(as.vector(train[row, -1]), nrow = 28, ncol = 28, byrow = TRUE)
img <- apply(img, 1:2, as.numeric)

# Rotate the image 90 degrees clockwise
rotated_img <- t(apply(img, 2, rev))

# Display the rotated image
image(rotated_img, main = paste("Row", row, ""))
