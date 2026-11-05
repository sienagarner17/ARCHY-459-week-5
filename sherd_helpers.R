# Helper functions for the module 5 lab (Tusayan White Ware sherds).
#
# The lesson loads this file with source("sherd_helpers.R"), after loading
# the tidyverse, torch, torchvision and magick packages. The lime plots
# also need the lime and patchwork packages, which the lesson loads in
# the grad section.
#
# Contents:
#   load_sherd(), prepare_sherd()   prepare images the same way as the paper
#   sherd_dataset()                 feed images and labels to torch
#   grad_cam(), plot_cam()          Grad-CAM heat maps
#   sherd_classifier() and methods  connect our model to the lime package
#   make_colour_copies(),
#   load_lime_images(), lime_pair() images and plots for lime


# Image preparation ----------------------------------------------------------

# Loads one sherd photo, converts it to grayscale, copies the gray channel
# three times (the pretrained model expects 3 colour channels), and
# resizes it to 224 x 224 pixels.
load_sherd <- function(path) {
  x <- base_loader(path)
  x <- transform_to_tensor(x)
  x <- transform_rgb_to_grayscale(x)
  x <- x$expand(c(3, -1, -1))
  transform_resize(x, c(224, 224))
}

# Loads a sherd, augments it if augment = TRUE, and rescales it.
#
# Augmentation rotates by a random angle up to 180 degrees and zooms in
# or out by up to 30%. torchvision fills the empty corners with black,
# so we invert the image (white background becomes black), rotate and
# zoom it, then invert it back, which leaves white corners.
#
# Rescaling matches the ImageNet photos the model was pretrained on:
# for each colour channel, subtract the ImageNet mean and divide by the
# ImageNet standard deviation.
prepare_sherd <- function(path, augment = FALSE) {
  x <- load_sherd(path)
  if (augment) {
    x <- 1 - transform_random_affine(1 - x, degrees = 180, scale = c(0.7, 1.3))
  }
  transform_normalize(x, mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}


# Torch dataset ----------------------------------------------------------

# Returns one prepared image and its type label at a time.
# The label is the type's position in the factor levels (1 to 8).
sherd_dataset <- dataset(
  initialize = function(df, augment) {
    self$paths <- df$path
    self$labels <- as.integer(df$type)
    self$augment <- augment
  },
  .getitem = function(i) {
    x <- prepare_sherd(self$paths[i], self$augment)
    list(x = x, y = self$labels[i])
  },
  .length = function() {
    length(self$paths)
  }
)


# Grad-CAM ----------------------------------------------------------------

# Returns a 224 x 224 matrix of Grad-CAM values (0 to 1) for one sherd
# and one type. `model` is our trained sherd_net, on the CPU.
#
# The last convolutional layer of ResNet18 produces 512 feature maps on a
# 7 x 7 grid. Grad-CAM weights each map by how much it raises the score
# for the chosen type (the gradient), adds up the weighted maps, keeps
# only the positive parts, and stretches the result to 224 x 224 pixels.
grad_cam <- function(path, type, model) {
  r <- model$resnet
  x <- prepare_sherd(path)
  x <- x$unsqueeze(1)

  # run the network up to its last convolutional layer and keep those feature maps
  maps <- r$maxpool(r$relu(r$bn1(r$conv1(x))))
  maps <- r$layer4(r$layer3(r$layer2(r$layer1(maps))))
  maps$retain_grad()

  # finish the forward pass and get the score for the chosen type
  scores <- r$fc(torch_flatten(r$avgpool(maps), start_dim = 2))
  scores[1, as.integer(type)]$backward()

  # weight each feature map by how much it raises that score
  weights <- maps$grad$mean(dim = c(3, 4), keepdim = TRUE)
  cam <- nnf_relu((weights * maps)$sum(dim = 2, keepdim = TRUE))
  cam <- nnf_interpolate(cam, size = c(224, 224), mode = "bilinear", align_corners = FALSE)
  cam <- as_array(cam[1, 1, , ])
  cam / max(cam)
}

# Draws the sherd and lays the Grad-CAM heat map for `type` over it.
plot_cam <- function(path, type, title, model) {
  cam <- grad_cam(path, type, model)
  sherd <- as_array(load_sherd(path)[1, , ])

  # one row per pixel, with the top row of the image at the top of the plot
  heat <- expand_grid(y = 224:1, x = 1:224) |>
    mutate(cam = as.vector(t(cam)))

  ggplot(heat, aes(x, y)) +
    annotation_raster(as.raster(sherd), xmin = 0.5, xmax = 224.5, ymin = 0.5, ymax = 224.5) +
    geom_raster(aes(fill = cam), alpha = 0.45) +
    scale_fill_viridis_c(option = "inferno", guide = "none") +
    coord_fixed(expand = FALSE) +
    labs(title = title) +
    theme_void()
}


# Connecting our model to lime --------------------------------------------

# lime needs an object it can ask for predictions. This one stores our
# torch model and the names of the 8 types, in the same order as the
# model's 8 outputs.
sherd_classifier <- function(model, type_levels) {
  structure(list(model = model, type_levels = type_levels), class = "sherd_classifier")
}

# Tells lime that our model is a classifier.
model_type.sherd_classifier <- function(x, ...) {
  "classification"
}

# Gives lime a data frame of probabilities: one row per image,
# one column per type.
predict_model.sherd_classifier <- function(x, newdata, type, ...) {
  scores <- with_no_grad(x$model(newdata))
  probs <- as_array(nnf_softmax(scores, dim = 2))
  colnames(probs) <- x$type_levels
  as.data.frame(probs)
}


# Images for lime ----------------------------------------------------------

# lime expects colour images, but the sherd photos are grayscale,
# so we save a colour copy of each one and return the copies' paths.
make_colour_copies <- function(paths) {
  map_chr(paths, \(p) {
    copy <- tempfile(fileext = ".png")
    image_read(p) |>
      image_convert(colorspace = "sRGB") |>
      image_write(copy, format = "png24")
    copy
  })
}

# lime hides patches by making them transparent. We paint transparent
# areas white, the same as the photo background, then prepare each image
# with prepare_sherd(), exactly as for the test sherds.
load_lime_images <- function(paths) {
  images <- map(paths, \(p) {
    flat <- tempfile(fileext = ".png")
    image_read(p) |>
      image_background("white") |>
      image_flatten() |>
      image_write(flat, format = "png24")
    prepare_sherd(flat)
  })
  torch_stack(images)
}


# lime plots ------------------------------------------------------------------

# Puts two lime plots side by side: the outlines of all 50 patches, and
# the 5 patches that most support the predicted type (shaded blue).
# explain() returns a tibble, which plot_image_explanation() cannot handle,
# so we change it to a plain data frame first. The lime weights for these
# sherds are below the default cut-off of 0.02, so threshold = 0 shows
# all 5 patches.
lime_pair <- function(path, explanation, title) {
  class(explanation) <- "data.frame"

  outlines <- plot_superpixels(path, n_superpixels = 50) +
    labs(title = title)
  supports <- plot_image_explanation(explanation, threshold = 0)

  (outlines + supports) &
    theme(axis.title.x = element_blank(), axis.title.y = element_blank())
}
