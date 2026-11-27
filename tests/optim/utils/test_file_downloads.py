#!/usr/bin/env python3
import unittest

import requests
import torch

BASE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/captum/"
BASE_URL_MODELS = "https://pytorch.s3.amazonaws.com/models/captum/"


class TestFileExistence(unittest.TestCase):
    def test_clip_bpe_simple_vocab_48895_txt(self) -> None:
        url = BASE_URL_MODELS + "clip_bpe_simple_vocab_48895.txt"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, f"File not found: {url}")
        lines = response.text.strip().split("\n")
        self.assertEqual(
            len(lines),
            48895,
            f"Unexpected number of lines in {url},"
            + f"expected 48895 but got {len(lines)}",
        )

    def test_inceptionv1_mixed4c_relu_samples_activations_pt(self) -> None:
        raise unittest.SkipTest()
        url = BASE_URL + "inceptionv1_mixed4c_relu_samples_activations.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_inceptionv1_mixed4c_relu_samples_attributions_pt(self) -> None:
        raise unittest.SkipTest()
        url = BASE_URL + "inceptionv1_mixed4c_relu_samples_attributions.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_inceptionv1_mixed5b_relu_samples_activations_pt(self) -> None:
        raise unittest.SkipTest()
        url = BASE_URL + "inceptionv1_mixed5b_relu_samples_activations.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_inceptionv1_mixed5b_relu_samples_attributions_pt(self) -> None:
        raise unittest.SkipTest()
        url = BASE_URL + "inceptionv1_mixed5b_relu_samples_attributions.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_inception5h_pth(self) -> None:
        url = BASE_URL_MODELS + "inception5h.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_inceptionv1_places365_pth(self) -> None:
        url = BASE_URL + "inceptionv1_places365.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_vgg16_caffe_features_pth(self) -> None:
        url = BASE_URL + "vgg16_caffe_features.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_clip_resnet50x4_facets_pt(self) -> None:
        url = BASE_URL_MODELS + "clip_resnet50x4_facets.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, list), f"Failed to load list from: {url}"
        )

    def test_clip_resnet50x4_image_pt(self) -> None:
        url = BASE_URL_MODELS + "clip_resnet50x4_image.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )

    def test_clip_resnet50x4_text_pt(self) -> None:
        url = BASE_URL_MODELS + "clip_resnet50x4_text.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, check_hash=False
        )
        self.assertTrue(
            isinstance(state_dict, dict), f"Failed to load state dict from: {url}"
        )
