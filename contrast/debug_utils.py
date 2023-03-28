import os
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.utils import save_image
from PIL import ImageDraw
from PIL import Image

from contrast.flow import flow_viz


def prepare_imgs(coord_q, coord_k):
    if isinstance(coord_q, tuple):
        idx, epoch = None, None
        coord_q, test_imgs = coord_q
        coord_k, test_imgs2 = coord_k
        if isinstance(test_imgs, list):
            test_imgs, idx, epoch = test_imgs
            test_imgs2, _, _ = test_imgs2
        ndim = test_imgs.ndim
        img1, img2 = None, None
        if ndim > 4:
            img1 = test_imgs[:, 1]
            img2 = test_imgs[:, 2]
            test_imgs = test_imgs[:, 0]
            test_imgs2 = None
    return coord_q, coord_k, test_imgs, test_imgs2, img1, img2, idx, epoch


def prepare_dirs(out_root_src, test_imgs, test_imgs2, coord_q, coord_k, idx, epoch, img1, img2, is_calc_flow, is_pos=False):
    base_name = os.path.basename(out_root_src)
    is_reverse = base_name == "2"
    color = [(255, 165, 0), (0, 0, 255)]
    if is_reverse:
        color_tmp = color[0]
        color[0] = color[1]
        color[1] = color_tmp
    out_root = f"{out_root_src}/epoch_{epoch}"
    out_path_center = f"{out_root}/center"
    out_path = f"{out_root}/no_center"
    os.makedirs(out_path_center, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    if test_imgs2 is not None:
        out_path_crop = f"{out_root}/crop"
        os.makedirs(out_path_crop, exist_ok=True)
        rec_imgs, crop_imgs = draw_rects(test_imgs, test_imgs2, coord_q, coord_k, color, idx, out_path_crop)
        test_imgs, img1, img2 = rec_imgs

    color = [(255, 0, 0), (0, 255, 0)]
    if is_reverse:
        color_tmp = color[0]
        color[0] = color[1]
        color[1] = color_tmp

    calc_flow_list = []

    if is_calc_flow:
        out_path_center_flo = f"{out_root}/center_flow"
        out_path_flo = f"{out_root}/no_center_flow"
        os.makedirs(out_path_center_flo, exist_ok=True)
        os.makedirs(out_path_flo, exist_ok=True)
        calc_flow_list = [out_path_flo, out_path_center_flo]

    out_path_pos = None
    if is_pos:
        out_path_pos = f"{out_root}/positive_pair"
        os.makedirs(out_path_pos, exist_ok=True)

    return out_path, out_path_center, color, test_imgs, img1, img2, calc_flow_list, out_path_pos


def calc_flow_grid_crop_size(coord_q, coord_k):
    q_grids, k_grids = [], []
    for (q_w, q_h), (k_w, k_h) in zip(coord_q[:, 6:8], coord_k[:, 6:8]):
        q_crop_w, q_crop_h = int(q_w.item()), int(q_h.item())
        k_crop_w, k_crop_h = int(k_w.item()), int(k_h.item())
        q_grid_tmp = torch.meshgrid(torch.arange(q_crop_h), torch.arange(q_crop_w))
        q_grid_tmp = torch.stack(q_grid_tmp[::-1]).to(coord_q.dtype).to(coord_q.device)
        q_grids.append(q_grid_tmp)
        k_grid_tmp = torch.meshgrid(torch.arange(k_crop_h), torch.arange(k_crop_w))
        k_grid_tmp = torch.stack(k_grid_tmp[::-1]).to(coord_k.dtype).to(coord_k.device)
        k_grids.append(k_grid_tmp)
    return q_grids, k_grids


def debug_print(q_start_x="None", q_start_y="None", k_start_x="None", k_start_y="None",
                q_bin_width="None", q_bin_height="None", k_bin_width="None",
                k_bin_height="None", q_grids="None", k_grids="None", q_bin_diag="None",
                k_bin_diag="None", max_bin_diag="None", center_q_x="None",
                center_q_y="None", center_k_x="None", center_k_y="None",
                center_q_x_o="None", center_q_y_o="None", center_k_x_o="None",
                center_k_y_o="None", q_flip_x="None", q_flip_y="None", k_flip_x="None",
                k_flip_y="None"):
    rank = torch.distributed.get_rank()
    if rank == 0:
        if q_flip_x != "None":
            print(f"q_flip_x: {q_flip_x.shape}", q_flip_x.tolist())
        if q_flip_y != "None":
            print(f"q_flip_y: {q_flip_y.shape}", q_flip_y.tolist())
        if k_flip_x != "None":
            print(f"k_flip_x: {k_flip_x.shape}", k_flip_x.tolist())
        if k_flip_y != "None":
            print(f"k_flip_y: {k_flip_y.shape}", k_flip_y.tolist())
        if q_start_x != "None":
            print(f"q_start_x: {q_start_x.shape}", q_start_x.tolist())
        if q_start_y != "None":
            print(f"q_start_y: {q_start_y.shape}", q_start_y.tolist())
        if k_start_x != "None":
            print(f"k_start_x: {k_start_x.shape}", k_start_x.tolist())
        if k_start_y != "None":
            print(f"k_start_y: {k_start_y.shape}", k_start_y.tolist())
        if q_bin_width != "None":
            print(f"q_bin_width: {q_bin_width.shape}", q_bin_width.tolist())
        if q_bin_height != "None":
            print(f"q_bin_height: {q_bin_height.shape}", q_bin_height.tolist())
        if k_bin_width != "None":
            print(f"k_bin_width: {k_bin_width.shape}", k_bin_width.tolist())
        if k_bin_height != "None":
            print(f"k_bin_height: {k_bin_height.shape}", k_bin_height.tolist())
        if q_grids != "None" and k_grids != "None":
            for ii, (q_grid, k_grid) in enumerate(zip(q_grids, k_grids)):
                if q_grid != "None":
                    print(f"{ii} q_grid: {q_grid.shape}")
                    # print(f"{ii} q_grid: {q_grid.shape}", q_grid.tolist())
                if k_grid != "None":
                    print(f"{ii} k_grid: {k_grid.shape}")
                    # print(f"{ii} k_grid: {k_grid.shape}", k_grid.tolist())
        if q_bin_diag != "None":
            print(f"q_bin_diag: {q_bin_diag.shape}", q_bin_diag.tolist())
        if k_bin_diag != "None":
            print(f"k_bin_diag: {k_bin_diag.shape}", k_bin_diag.tolist())
        if max_bin_diag != "None":
            print(f"max_bin_diag: {max_bin_diag.shape}", max_bin_diag.tolist())
        if center_q_x != "None":
            print(f"center_q_x: {center_q_x.shape}", center_q_x.tolist())
        if center_q_y != "None":
            print(f"center_q_y: {center_q_y.shape}", center_q_y.tolist())
        if center_k_x != "None":
            print(f"center_k_x: {center_k_x.shape}", center_k_x.tolist())
        if center_k_y != "None":
            print(f"center_k_y: {center_k_y.shape}", center_k_y.tolist())
        if center_q_x_o != "None":
            print(f"center_q_x_o: {center_q_x_o.shape}", center_q_x_o.tolist())
        if center_q_y_o != "None":
            print(f"center_q_y_o: {center_q_y_o.shape}", center_q_y_o.tolist())
        if center_k_x_o != "None":
            print(f"center_k_x_o: {center_k_x_o.shape}", center_k_x_o.tolist())
        if center_k_y_o != "None":
            print(f"center_k_y_o: {center_k_y_o.shape}", center_k_y_o.tolist())


def calc_grid_no_center(x_array, y_array, q_bin_width, q_bin_height, k_bin_width,
                        k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                        W_orig, H_orig):
    q_x = x_array * q_bin_width + q_start_x
    q_y = y_array * q_bin_height + q_start_y
    k_x = x_array * k_bin_width + k_start_x
    k_y = y_array * k_bin_height + k_start_y
    q_x = q_x * (W_orig - 1)
    q_y = q_y * (H_orig - 1)
    k_x = k_x * (W_orig - 1)
    k_y = k_y * (H_orig - 1)
    return q_x, q_y, k_x, k_y


def load_img_from_pil(img):
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


# concat width
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# concat height
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def draw_rect_simple(img_src, rects, colors):
    if isinstance(img_src, torch.Tensor):
        img = img_src.clone()
        img = transforms.ToPILImage(mode="RGB")(img)
    else:
        img = img_src.copy()
    # rectcolor = (255, 0, 0)  # red
    linewidth = 4  # 線の太さ

    if len(rects) > 0:
        draw = ImageDraw.Draw(img)  # 準備

    for i, (rect, rectcolor) in enumerate(zip(rects, colors)):
        rect_tup = [(rect[0], rect[1]), (rect[2], rect[3])]
        draw.rectangle(rect_tup, outline=rectcolor, width=linewidth)

    return img


def draw_rects(img1, img2, coord_q, coord_k, colors, idx=0, out_root="./"):
    rank = torch.distributed.get_rank()
    view1 = coord_q.clone()
    view1 = view1[:, 4:8].to(torch.int)
    view1[:, 2] = view1[:, 2] + view1[:, 0]
    view1[:, 3] = view1[:, 3] + view1[:, 1]
    view1 = view1.tolist()
    view2 = coord_k.clone()
    view2 = view2[:, 4:8].to(torch.int)
    view2[:, 2] = view2[:, 2] + view2[:, 0]
    view2[:, 3] = view2[:, 3] + view2[:, 1]
    view2 = view2.tolist()
    orig_im1 = img1.clone().to(torch.uint8)
    orig_im2 = img2.clone().to(torch.uint8)
    out_img1_img2, out_img1, out_img2 = [], [], []
    crop_imgs = []
    for b_idx, (im1, im2, v1, v2) in enumerate(zip(orig_im1, orig_im2, view1, view2)):
        pil_im1 = draw_rect_simple(im1, [], [])
        pil_im2 = draw_rect_simple(im2, [], [])
        img = draw_rect_simple(im2, [v1, v2], colors)
        img1 = draw_rect_simple(im1, [v1], [colors[0]])
        img2 = draw_rect_simple(im2, [v2], [colors[1]])
        img.save(os.path.join(out_root, f"crop_rec_img_{idx}_{b_idx}_{rank}.png"))
        img1.save(os.path.join(out_root, f"crop_rec_img_1_{idx}_{b_idx}_{rank}.png"))
        img2.save(os.path.join(out_root, f"crop_rec_img_2_{idx}_{b_idx}_{rank}.png"))
        out_img1_img2.append(load_img_from_pil(img))
        out_img1.append(load_img_from_pil(img1))
        out_img2.append(load_img_from_pil(img2))
        im1_crop = pil_im1.crop(v1)
        im2_crop = pil_im2.crop(v2)
        im1_crop.save(os.path.join(out_root, f"crop_img_1_{idx}_{b_idx}_{rank}.png"))
        im2_crop.save(os.path.join(out_root, f"crop_img_2_{idx}_{b_idx}_{rank}.png"))
        crop_imgs.append([load_img_from_pil(im1_crop), load_img_from_pil(im2_crop)])

    out_img1_img2 = torch.stack(out_img1_img2)
    out_img1 = torch.stack(out_img1)
    out_img2 = torch.stack(out_img2)

    return [out_img1_img2, out_img1, out_img2], crop_imgs
    # return [out_img1_img2, out_img1_img2.clone(), out_img1_img2.clone()], crop_imgs


def create_colors(color_s=None, num=None, rgb="rgb"):
    def color_code_to_rgb(color_code, rgb="rgb"):
        if len(rgb) != 3:
            rgb = "rgb"
        if "r" not in rgb or "g" not in rgb or "b" not in rgb:
            rgb = "rgb"
        h = hex(color_code)
        h = h[2:]
        h = h.zfill(6)
        color = []
        for i in range(0, 6, 2):
            c = int(h[i:i+2], 16)
            color.append(c)
        ans_color = color.copy()
        for i, s in enumerate(rgb):
            if s == "r":
                ans_color[0] = color[i]
            elif s == "g":
                ans_color[1] = color[i]
            elif s == "b":
                ans_color[2] = color[i]
        return tuple(ans_color)

    def rgb_to_hex(rgb):
        r, g, b = rgb
        r, g, b = hex(r), hex(g), hex(b)
        c = r[2:].zfill(2) + g[2:].zfill(2) + b[2:].zfill(2)
        return int(c, 16)

    if num is None:
        num = 49
    if color_s is None:
        color_code = 1250067
    else:
        color_code = rgb_to_hex(color_s)

    color_list = []
    for i in range(num):
        color_code += 3500
        # color_code += 150
        if color_code > 16250870:
            color_code -= 15527148
        c = color_code_to_rgb(color_code, rgb)
        color_list.append(c)

    return color_list


def draw_point_simple(img_src, points_src, color_list, width=4, bin_width=None, bin_height=None):
    if isinstance(img_src, torch.Tensor):
        img = img_src.clone()
        img = transforms.ToPILImage(mode="RGB")(img)
    else:
        img = img_src.copy()

    if isinstance(points_src, torch.Tensor):
        points = points_src.clone()
        ndim = points.ndim
        if ndim >= 3:
            points = list(itertools.chain.from_iterable(points.tolist()))
            if width <= 0:
                points = list(itertools.chain.from_iterable(points))
        elif ndim == 1:
            points = points.unsqueeze(0).tolist()
        else:
            points = points.tolist()
    else:
        points = points_src

    # rectcolor = (255, 0, 0)  # red
    # width = 4  # 線の太さ

    if len(points) > 0:
        draw = ImageDraw.Draw(img)  # 準備

    if width <= 0:
        draw.point(points, fill=color_list)
        return img

    len_c = len(color_list)
    if isinstance(color_list, list):
        print(len_c, "# color")
    for i, point in enumerate(points):
        if isinstance(color_list, tuple):
            l_color = color_list
        else:
            l_color = tuple(color_list[i % len_c])
        point_tmp = [(point[0] - (width / 2), point[1] - (width / 2)), (point[0] + (width / 2), point[1] + (width / 2))]
        draw.ellipse(point_tmp, fill=l_color, width=width)
        if bin_width is not None and bin_height is not None:
            rect_1 = (int(point[0] - (bin_width.item() / 2)), int(point[1] - (bin_height.item() / 2)))
            rect_2 = (int(point[0] + (bin_width.item() / 2)), int(point[1] + (bin_height.item() / 2)))
            rect_tup = [rect_1, rect_2]
            draw.rectangle(rect_tup, outline=l_color, width=width)

    return img


def make_grid(x, y):
    ndim = x.ndim
    assert ndim == 3 or ndim == 2
    if ndim == 3:
        grids = torch.stack([x, y]).permute(1, 0, 2, 3)
    elif ndim == 2:
        grids = torch.stack([x, y]).unsqueeze(0)

    return grids


def adjust_img_dim(img_src):
    img_ndim = img_src.ndim
    assert img_ndim == 4 or img_ndim == 3
    img = img_src.clone()
    if img_ndim == 3:
        img = img.unsqueeze(0)
    return img


def adjust_dim(tar, out_dim=4):
    if tar is None:
        return None
    assert isinstance(out_dim, int)
    assert isinstance(tar, torch.Tensor)
    in_dim = tar.ndim
    num = out_dim - in_dim
    out = tar.clone()
    if num == 0:
        return out
    is_down_dim = num <= 0
    for i in range(abs(num)):
        if is_down_dim:
            assert out.shape[0] == 1
            out = out.squeeze(0)
        else:
            out = out.unsqueeze(0)
    return out


def apply_mask(img_src, mask_src=None, pad_v=255):
    if mask_src is None or img_src is None:
        return None
    in_dim = img_src.ndim
    mask = adjust_dim(mask_src, 3)
    img = adjust_dim(img_src, 4)
    mask_rev = torch.logical_not(mask)
    img_mask = img.clone()
    img_mask = img_mask.permute(0, 2, 3, 1)
    img_mask[mask_rev] = pad_v
    img_mask = img_mask.permute(0, 3, 1, 2)
    img_mask = adjust_dim(img_mask, in_dim)
    return img_mask


def save_img_tensor(img_tensor, out_root, fname, ext="png"):
    assert img_tensor is not None
    assert out_root is not None
    assert fname is not None and fname != ""
    assert isinstance(img_tensor, torch.Tensor)
    ndim = img_tensor.ndim
    assert ndim == 3 or ndim == 4
    out_name = os.path.join(out_root, fname)
    img_copy = adjust_dim(img_tensor, 4)
    num = img_copy.shape[0]
    for i, img in enumerate(img_copy):
        out_img = transforms.ToPILImage(mode="RGB")(img)
        l_out_name = out_name if num == 1 else f"{out_name}_{i}"
        out_img.save(f"{l_out_name}.{ext}")


def draw_points_onegrid(x, y, img_src, out_path, color, name="plot_point", width=4, bin_width=None, bin_height=None):
    # print(color, "color in one grid")
    if isinstance(color, list):
        color = color[0]
    # print(color, "color in one grid")

    grids = make_grid(x, y)
    test_imgs = adjust_img_dim(img_src)

    if bin_width is None:
        nb = len(test_imgs)
        bin_width = [None] * nb
        bin_height = [None] * nb

    for idx, (orig_im, grid) in enumerate(zip(test_imgs, grids)):
        img = draw_point_simple(orig_im, grid.permute(1, 2, 0), color, width, bin_width[idx], bin_height[idx])
        img.save(os.path.join(out_path, f"{name}_{idx}.png"))


def draw_points(q_x, q_y, k_x, k_y, img_src, out_path, color, name="plot_point", width=4,
                q_bin_width=None, q_bin_height=None, k_bin_width=None, k_bin_height=None):
    q_grids = make_grid(q_x, q_y)
    k_grids = make_grid(k_x, k_y)
    test_imgs = adjust_img_dim(img_src)

    if q_bin_width is None:
        nb = len(test_imgs)
        q_bin_width = [None] * nb
        q_bin_height = [None] * nb
        k_bin_width = [None] * nb
        k_bin_height = [None] * nb

    for idx, (orig_im, q_grid, k_grid) in enumerate(zip(test_imgs, q_grids, k_grids)):
        img = draw_point_simple(orig_im, q_grid.permute(1, 2, 0), color[0], width, q_bin_width[idx], q_bin_height[idx])
        img = draw_point_simple(img, k_grid.permute(1, 2, 0), color[1], width, k_bin_width[idx], k_bin_height[idx])
        img.save(os.path.join(out_path, f"{name}_{idx}.png"))


def draw_point_positive_pair(q_x, q_y, k_x, k_y, q_x_final, q_y_final, k_x_final,
                             k_y_final, img1_src, img2_src, out_path, color_s,
                             pos_masks, name="plot_point_positive", width=4,
                             q_bin_width=None, k_bin_width=None, q_bin_height=None,
                             k_bin_height=None):
    q_grids = make_grid(q_x, q_y)
    k_grids = make_grid(k_x, k_y)
    q_grids_final = make_grid(q_x_final, q_y_final)
    k_grids_final = make_grid(k_x_final, k_y_final)
    im1 = adjust_img_dim(img1_src)
    im2 = adjust_img_dim(img2_src)
    nb, c, h, w = q_grids.shape
    q_grids = q_grids.permute(0, 2, 3, 1)
    k_grids = k_grids.permute(0, 2, 3, 1)
    q_grids = q_grids.view(nb, h * w, 1, c).repeat(1, 1, h * w, 1)
    k_grids = k_grids.view(nb, 1, h * w, c).repeat(1, h * w, 1, 1)
    q_grids_final = q_grids_final.permute(0, 2, 3, 1)
    k_grids_final = k_grids_final.permute(0, 2, 3, 1)
    q_grids_final = q_grids_final.view(nb, h * w, 1, c).repeat(1, 1, h * w, 1)
    k_grids_final = k_grids_final.view(nb, 1, h * w, c).repeat(1, h * w, 1, 1)
    color_list = create_colors(color_s[0])
    # color_list = color_s[0]

    for idx, (orig_im1, orig_im2, q_grid, k_grid, pos_mask) in enumerate(zip(im1, im2, q_grids, k_grids, pos_masks)):
        q_grid_final = q_grids_final[idx]
        k_grid_final = k_grids_final[idx]
        l_out_path = os.path.join(out_path, f"batch_{idx}")
        os.makedirs(l_out_path, exist_ok=True)

        img1_all = draw_point_simple(orig_im1, q_grid[pos_mask], color_list, width, q_bin_width[idx], q_bin_height[idx])
        img2_all = draw_point_simple(orig_im2, k_grid[pos_mask], color_list, width, k_bin_width[idx], k_bin_height[idx])
        img_all = get_concat_h(img1_all, img2_all)
        img_all.save(os.path.join(out_path, f"{name}_{idx}.png"))

        def get_tmp_color(color):
            if color[0] == 255:
                color_tmp = (255, 153, 153)
            elif color[1] == 255:
                color_tmp = (153, 255, 153)
            elif color[2] == 255:
                color_tmp = (153, 153, 255)
            return color_tmp

        color_tmp = [get_tmp_color(c) for c in color_s]

        for jdx, (q_g, k_g, p_mask) in enumerate(zip(q_grid, k_grid, pos_mask)):
            l_q_grid = q_g[0]
            l_k_grid = k_g[p_mask]
            l_q_grid_final = q_grid_final[jdx][0]
            l_k_grid_final = k_grid_final[jdx][p_mask]
            img1 = draw_point_simple(orig_im1, l_q_grid, color_s[0], width, q_bin_width[idx], q_bin_height[idx])
            img2 = draw_point_simple(orig_im2, l_k_grid, color_s[0], width, k_bin_width[idx], k_bin_height[idx])
            img3 = draw_point_simple(orig_im2, l_q_grid, color_tmp[0], width, q_bin_width[idx], q_bin_height[idx])
            img4 = draw_point_simple(orig_im2, l_k_grid, color_tmp[1], width, k_bin_width[idx], k_bin_height[idx])
            img3 = draw_point_simple(img3, l_q_grid_final, color_s[0], width, q_bin_width[idx], q_bin_height[idx])
            img4 = draw_point_simple(img4, l_k_grid_final, color_s[1], width, k_bin_width[idx], k_bin_height[idx])
            img3 = draw_point_simple(img3, l_k_grid_final, color_s[1], width, k_bin_width[idx], k_bin_height[idx])
            img4 = draw_point_simple(img4, l_q_grid_final, color_s[0], width, q_bin_width[idx], q_bin_height[idx])
            # img1 = draw_point_simple(orig_im1, l_q_grid[0], color_s[0], width, q_bin_width[idx], q_bin_height[idx])
            # img2 = draw_point_simple(orig_im2, l_k_grid[0], color_s[0], width, k_bin_width[idx], k_bin_height[idx])
            img = get_concat_h(img1, img2)
            img_tmp = get_concat_h(img3, img4)
            img = get_concat_v(img, img_tmp)
            img.save(os.path.join(l_out_path, f"{name}_{jdx}.png"))


def warp(im2, flo, mask=None, out_root="./", name="flow_warp_img2_to_img1"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    im2: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = im2.size()
    im2_copy = im2.clone()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flo.is_cuda or im2_copy.is_cuda:
        if flo.is_cuda:
            im2_copy = im2_copy.to(flo.device)
            grid = grid.to(flo.device)
        else:
            flo = flo.to(im2_copy.device)
            grid = grid.to(im2_copy.device)

    im2_copy = im2_copy.to(grid.dtype)

    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(im2_copy - 255, vgrid, align_corners=True)
    output = output + 255
    output = output.detach().cpu().to(torch.uint8)

    save_img_tensor(output, out_root, f"{name}_orig")

    if mask is not None:
        out_mask = apply_mask(output, mask, 255)
        save_img_tensor(out_mask, out_root, f"{name}_mask")
    else:
        out_mask = None
    # mask = torch.ones(im2.size()).to(DEVICE)
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    return output, out_mask


def draw_warp_img(grid_x, grid_y, W_orig, H_orig, img_src, name, out_path, mask_src=None):
    grid = torch.stack([grid_x, grid_y]).unsqueeze(0)
    grid[:, 0] = grid[:, 0] / (W_orig - 1)
    grid[:, 1] = grid[:, 1] / (H_orig - 1)
    grid[:, 0] = 2 * grid[:, 0] - 1
    grid[:, 1] = 2 * grid[:, 1] - 1
    # warp img2 -> img1
    orig_img_copy = img_src.clone()
    orig_img_copy = orig_img_copy.to(grid.device).to(grid.dtype)
    orig_img_copy = orig_img_copy.unsqueeze(0)
    of_img = F.grid_sample(orig_img_copy, grid.permute(0, 2, 3, 1),
                           align_corners=True)
    of_img = of_img.detach().cpu()
    orig_img_copy = orig_img_copy.detach().cpu()
    orig_img_copy = orig_img_copy.to(torch.uint8)
    of_img = of_img.to(torch.uint8)
    # save_img_tensor(of_img, out_path, name)
    # save_image(orig_img_copy, os.path.join(out_path, f"{name}0.png"))
    save_img_tensor(orig_img_copy, out_path, f"{name}_orig")
    # save_image(of_img, os.path.join(out_path, f"{name}.png"))
    save_img_tensor(of_img, out_path, name)

    if mask_src is not None:
        mask = mask_src.clone()
        if mask.ndim == 4:
            nb, c, h, w = mask.shape
            mask = mask.reshape(nb, h, w)
        of_img_copy = apply_mask(of_img, mask, 255)
        save_img_tensor(of_img_copy, out_path, f"{name}_mask")
        # orig_img_copy = apply_mask(orig_img_copy, mask, 0.0)
        # save_img_tensor(orig_img_copy, out_path, f"{name}_orig_mask")
    # save_image((of_img * 255), os.path.join(out_path, f"{name}2.png"))
    # rank = torch.distributed.get_rank()
    # if rank == 0:
    #     print(f"{i} orig_img_copy: {orig_img_copy.shape}", orig_img_copy.tolist())
    #     print(f"{i} of_q_grid: {of_q_grid.shape}", of_q_grid.tolist())
    #     print(f"{i} of_img: {of_img.shape}", of_img.tolist())


def draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
                    k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                    W, H, W_orig, H_orig, test_imgs, img1, img2, out_path, color,
                    name, is_center=False, flow_fwd=None, out_path_flo="",
                    add_optical_flow=None, mask=None):
    is_calc_flow = flow_fwd is not None and add_optical_flow is not None
    size = (H_orig, W_orig)
    # rank = torch.distributed.get_rank()
    for i, (q_grid, k_grid) in enumerate(zip(q_grids, k_grids)):
        q_crop_h, q_crop_w = q_grid.shape[-2:]
        k_crop_h, k_crop_w = k_grid.shape[-2:]
        l_q_bin_width = q_bin_width[i] * W / (q_crop_w)
        l_q_bin_height = q_bin_height[i] * H / (q_crop_h)
        l_k_bin_width = k_bin_width[i] * W / (k_crop_w)
        l_k_bin_height = k_bin_height[i] * H / (k_crop_h)
        # l_q_bin_width = q_bin_width[i] * W / (q_crop_w - 1)
        # l_q_bin_height = q_bin_height[i] * H / (q_crop_h - 1)
        # l_k_bin_width = k_bin_width[i] * W / (k_crop_w - 1)
        # l_k_bin_height = k_bin_height[i] * H / (k_crop_h - 1)
        if is_center:
            center_q_x_tmp = (q_grid[0] + 0.5) * l_q_bin_width + q_start_x[i]
            center_q_y_tmp = (q_grid[1] + 0.5) * l_q_bin_height + q_start_y[i]
            # center_q_x_tmp = (q_grid[0] + 0.5) * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h)) + (q_start_y[i] * (H_orig - 1))
            # center_q_x_tmp = (q_grid[0] + 0.5) * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w - 1)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h - 1)) + (q_start_y[i] * (H_orig - 1))
            center_k_x_tmp = (k_grid[0] + 0.5) * l_k_bin_width + k_start_x[i]
            center_k_y_tmp = (k_grid[1] + 0.5) * l_k_bin_height + k_start_y[i]
            # center_k_x_tmp = (k_grid[0] + 0.5) * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = (k_grid[1] + 0.5) * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h)) + (k_start_y[i] * (H_orig - 1))
            # center_k_x_tmp = (k_grid[0] + 0.5) * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w - 1)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h - 1)) + (k_start_y[i] * (H_orig - 1))
        else:
            center_q_x_tmp = q_grid[0] * l_q_bin_width + q_start_x[i]
            center_q_y_tmp = q_grid[1] * l_q_bin_height + q_start_y[i]
            # center_q_x_tmp = q_grid[0] * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = q_grid[1] * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h)) + (q_start_y[i] * (H_orig - 1))
            # center_q_x_tmp = q_grid[0] * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w - 1)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = q_grid[1] * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h - 1)) + (q_start_y[i] * (H_orig - 1))
            center_k_x_tmp = k_grid[0] * l_k_bin_width + k_start_x[i]
            center_k_y_tmp = k_grid[1] * l_k_bin_height + k_start_y[i]
            # center_k_x_tmp = k_grid[0] * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = k_grid[1] * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h)) + (k_start_y[i] * (H_orig - 1))
            # center_k_x_tmp = k_grid[0] * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w - 1)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = k_grid[1] * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h - 1)) + (k_start_y[i] * (H_orig - 1))

        center_q_x_tmp = center_q_x_tmp * (W_orig - 1)
        center_q_y_tmp = center_q_y_tmp * (H_orig - 1)
        center_k_x_tmp = center_k_x_tmp * (W_orig - 1)
        center_k_y_tmp = center_k_y_tmp * (H_orig - 1)

        draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path, color, f"{name}_{i}", 0)
        draw_points_onegrid(center_q_x_tmp, center_q_y_tmp, test_imgs[i], out_path, color[0], f"{name}_1grid_on_2frame_{i}", 0)
        draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path, color[1], f"{name}_2grid_on_2frame_{i}", 0)
        if img1 is not None:
            # draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, img1[i], out_path, color, f"{name}_1frame_{i}", 0)
            draw_points_onegrid(center_q_x_tmp, center_q_y_tmp, img1[i], out_path, color[0], f"{name}_1frame_{i}", 0)
        if img2 is not None:
            # draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, img2[i], out_path, color, f"{name}_2frame_{i}", 0)
            draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, img2[i], out_path, color[1], f"{name}_2frame_{i}", 0)
        # if rank == 0:
        #     print(f"{i} center_q_x_tmp: {center_q_x_tmp.shape}", center_q_x_tmp.tolist())
        #     print(f"{i} center_q_y_tmp: {center_q_y_tmp.shape}", center_q_x_tmp.tolist())
        #     print(f"{i} center_k_x_tmp: {center_k_x_tmp.shape}", center_k_x_tmp.tolist())
        #     print(f"{i} center_k_y_tmp: {center_k_y_tmp.shape}", center_k_x_tmp.tolist())
        #     print(f"{i} q_grid: {q_grid.shape}", q_grid.tolist())
        #     print(f"{i} k_grid: {k_grid.shape}", k_grid.tolist())

        if is_calc_flow:
            l_flow_fwd = flow_fwd[i].unsqueeze(0)
            l_mask = None if mask is None else mask[i].unsqueeze(0)
            l_center_q_x_tmp = center_q_x_tmp.unsqueeze(0)
            l_center_q_y_tmp = center_q_y_tmp.unsqueeze(0)
            q_x_tmp, q_y_tmp, mask_fwd = add_optical_flow(l_flow_fwd, l_center_q_x_tmp,
                                                          l_center_q_y_tmp, size,
                                                          l_mask)
            q_x_tmp = q_x_tmp.squeeze(0)
            q_y_tmp = q_y_tmp.squeeze(0)
            draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path_flo, color, f"{name}_{i}", 0)
            draw_points_onegrid(center_q_x_tmp, center_q_y_tmp, test_imgs[i], out_path_flo, color[0], f"{name}_1frame_on_2frame_{i}", 0)
            draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path_flo, color[1], f"{name}_2frame_on_2frame_{i}", 0)
            draw_warp_img(center_q_x_tmp, center_q_y_tmp, W_orig, H_orig, test_imgs[i],
                          f"{name}_2frame_aug_img_q_{i}", out_path_flo, mask_fwd)
            draw_warp_img(center_k_x_tmp, center_k_y_tmp, W_orig, H_orig, test_imgs[i],
                          f"{name}_2frame_aug_img_k_{i}", out_path_flo)
            if img1 is not None:
                # draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, img1[i], out_path_flo, color, f"{name}_1frame_{i}", 0)
                draw_points_onegrid(q_x_tmp, q_y_tmp, img1[i], out_path_flo, color[0], f"{name}_1frame_{i}", 0)
                draw_warp_img(center_q_x_tmp, center_q_y_tmp, W_orig, H_orig, img1[i],
                              f"{name}_1frame_aug_img_{i}", out_path_flo, mask_fwd)
            if img2 is not None:
                # draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, img2[i],
                #             out_path_flo, color, f"{name}_2frame_{i}", 0)
                draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, img2[i], out_path_flo, color[1], f"{name}_2frame_{i}", 0)
                draw_warp_img(center_k_x_tmp, center_k_y_tmp, W_orig, H_orig, img2[i],
                              f"{name}_2frame_aug_img_{i}", out_path_flo)
            # if rank == 0:
            #     print(f"{i} q_x_tmp: {q_x_tmp.shape}", q_x_tmp.tolist())
            #     print(f"{i} q_y_tmp: {q_y_tmp.shape}", q_x_tmp.tolist())
            #     print(f"{i} center_k_x_tmp: {center_k_x_tmp.shape}", center_k_x_tmp.tolist())
            #     print(f"{i} center_k_y_tmp: {center_k_y_tmp.shape}", center_k_x_tmp.tolist())
            draw_warp_img(q_x_tmp, q_y_tmp, W_orig, H_orig, test_imgs[i],
                          f"{name}_2frame_to_1frame_img_{i}_flo", out_path_flo,
                          mask_fwd)


def main_debug_calc_grid(q_start_x, q_start_y, k_start_x, k_start_y,
                         q_bin_width, q_bin_height, k_bin_width, k_bin_height, q_grids,
                         k_grids, q_x, q_y, k_x, k_y, test_imgs, img1, img2,
                         out_path, color, W_orig, H_orig,
                         q_x_flow=None, q_y_flow=None, k_x_flow=None, k_y_flow=None,
                         flow_fwd=None, out_path_flo="", add_optical_flow=None,
                         mask=None, name="plot_point", is_center=True):
    H, W = q_x.shape[-2:]
    is_plot_flow = q_x_flow is not None
    is_warp_flow = flow_fwd is not None
    int_q_bin_width = q_bin_width * (W_orig - 1)
    int_q_bin_height = q_bin_height * (H_orig - 1)
    int_k_bin_width = k_bin_width * (W_orig - 1)
    int_k_bin_height = k_bin_height * (H_orig - 1)

    def vizes(flos, name_viz=f"{name}_flo_cycle", mask_src=mask):
        def viz(flo, img1, name, mask=None):
            is_cat = img1 is not None
            if is_cat:
                img = img1.clone()
                img = img.permute(1, 2, 0).cpu().numpy()
            flo = flo.permute(1, 2, 0).cpu().numpy()

            # map flow to rgb image
            flo = flow_viz.flow_to_image(flo)

            # # cv2 method(numpy)
            # if is_cat:
            #     img_flo = np.concatenate([img, flo], axis=0)

            # import matplotlib.pyplot as plt
            # plt.imshow(img_flo / 255.0)
            # plt.show()
            # if name != "":
            #     fname = os.path.join(out_path_flo, f"{name}_cv2.png")
            #     if is_cat:
            #         cv2.imwrite(fname, img_flo[:, :, [2, 1, 0]])
            #     else:
            #         cv2.imwrite(fname, flo[:, :, [2, 1, 0]])

            # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
            # cv2.waitKey()

            out_name = os.path.join(out_path_flo, f"{name}.png")
            flo_img = torch.from_numpy(flo).permute(2, 0, 1)

            flo_img_mask = apply_mask(flo_img, mask, 0)
            if flo_img_mask is not None:
                out_name_mask = os.path.join(out_path_flo, f"{name}_mask.png")
                flo_img_mask_pil = transforms.ToPILImage(mode="RGB")(flo_img_mask)
                if is_cat:
                    img_pil_copy = img1.clone()
                    img_pil_copy = transforms.ToPILImage(mode="RGB")(img_pil_copy)
                    flo_img_mask_pil = get_concat_v(img_pil_copy, flo_img_mask_pil)
                flo_img_mask_pil.save(out_name_mask)

            flo_img_tmp = flo_img.clone()
            flo_img_tmp = flo_img_tmp.detach().cpu().to(torch.uint8)
            flo_img_tmp_img = transforms.ToPILImage(mode="RGB")(flo_img_tmp)
            if is_cat:
                img_pil = img1.clone()
                img_pil = transforms.ToPILImage(mode="RGB")(img_pil)
                flo_img_tmp_img = get_concat_v(img_pil, flo_img_tmp_img)
            flo_img_tmp_img.save(out_name)
            return flo_img, flo_img_mask

        if flos is None:
            return None

        flos_copy = adjust_dim(flos, 4)
        nb = flos.shape[0]
        img1_copy = adjust_dim(img1, 4)
        if img1_copy is None:
            img1_copy = [None] * nb
        if mask_src is None:
            mask_src = [None] * nb
        flo_img_list, flo_img_list_mask = [], []
        for i, (l_flo, l_img, l_mask) in enumerate(zip(flos_copy, img1_copy, mask_src)):
            l_flo_img, l_flo_img_mask = viz(l_flo, l_img, f"{name_viz}_{i}", l_mask)
            flo_img_list.append(l_flo_img)
            if l_flo_img_mask is not None:
                flo_img_list_mask.append(l_flo_img_mask)
        flo_imgs = torch.stack(flo_img_list)
        if len(flo_img_list_mask) > 0:
            flo_imgs_mask = torch.stack(flo_img_list_mask)
        else:
            flo_imgs_mask = None
        return flo_imgs, flo_imgs_mask

    if mask is not None and isinstance(mask, list):
        mask, flo_cycle = mask

    draw_points(q_x, q_y, k_x, k_y, test_imgs, out_path, color, f"{name}", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
    draw_points_onegrid(q_x, q_y, test_imgs, out_path, color[0], f"{name}_1frame_on_2frame", 4, int_q_bin_width, int_q_bin_height)
    if img1 is not None:
        draw_points(q_x, q_y, k_x, k_y, img1, out_path, color, f"{name}_on_1frame", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
        draw_points_onegrid(q_x, q_y, img1, out_path, color[0], f"{name}_1frame", 4, int_q_bin_width, int_q_bin_height)
    if img2 is not None:
        draw_points(q_x, q_y, k_x, k_y, img2, out_path, color, f"{name}_on_2frame", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
        draw_points_onegrid(k_x, k_y, img2, out_path, color[1], f"{name}_2frame", 4, int_k_bin_width, int_k_bin_height)

    if is_plot_flow:
        draw_points(q_x_flow, q_y_flow, k_x_flow, k_y_flow, test_imgs, out_path_flo, color, f"{name}", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
        # draw_points(q_x_flow[mask], q_y_flow[mask], k_x_flow, k_y_flow, test_imgs, out_path_flo, color)
        draw_points_onegrid(q_x_flow, q_y_flow, test_imgs, out_path_flo, color[0], f"{name}_1frame_on_2frame", 4, int_q_bin_width, int_q_bin_height)
        if img1 is not None:
            draw_points(q_x_flow, q_y_flow, k_x_flow, k_y_flow, img1, out_path_flo, color, f"{name}_on_1frame", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points_onegrid(q_x_flow, q_y_flow, img1, out_path_flo, color[0], f"{name}_1frame", 4, int_q_bin_width, int_q_bin_height)
        if img2 is not None:
            draw_points(q_x_flow, q_y_flow, k_x_flow, k_y_flow, img2, out_path_flo, color, f"{name}_on_2frame", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points_onegrid(k_x_flow, k_y_flow, img2, out_path_flo, color[1], f"{name}_2frame", 4, int_k_bin_width, int_k_bin_height)
        if is_warp_flow:
            flo_img, flo_img_mask = warp(test_imgs, flow_fwd, mask, out_path_flo)
            vizes(flo_cycle, f"{name}_flo_cycle", None)
            vizes(flow_fwd, f"{name}_flo_fwd", mask)
            draw_points(q_x_flow, q_y_flow, k_x_flow, k_y_flow, flo_img, out_path_flo, color, f"{name}_on_flo_img", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points(q_x_flow, q_y_flow, k_x_flow, k_y_flow, flo_img_mask, out_path_flo, color, f"{name}_on_flo_img_mask", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points_onegrid(q_x_flow, q_y_flow, flo_img, out_path_flo, color[0], f"{name}_1frame_on_2frame_on_flo_img", 4, int_q_bin_width, int_q_bin_height)
            draw_points_onegrid(q_x_flow, q_y_flow, flo_img_mask, out_path_flo, color[0], f"{name}_1frame_on_2frame_on_flo_img_mask", 4, int_q_bin_width, int_q_bin_height)
            draw_points(q_x, q_y, k_x, k_y, flo_img, out_path_flo, color, f"{name}_on_flo_img_noflowgrid", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points(q_x, q_y, k_x, k_y, flo_img_mask, out_path_flo, color, f"{name}_on_flo_img_mask_noflowgrid", 4, int_q_bin_width, int_q_bin_height, int_k_bin_width, int_k_bin_height)
            draw_points_onegrid(q_x, q_y, flo_img, out_path_flo, color[0], f"{name}_1frame_on_2frame_on_flo_img_noflowgrid", 4, int_q_bin_width, int_q_bin_height)
            draw_points_onegrid(q_x, q_y, flo_img_mask, out_path_flo, color[0], f"{name}_1frame_on_2frame_on_flo_img_mask_noflowgrid", 4, int_q_bin_width, int_q_bin_height)

    # check optical flow
    # draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
    #                 k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
    #                 W, H, W_orig, H_orig, test_imgs, img1, img2, out_path_center,
    #                 color, f"of_{name}", is_center)
    draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
                    k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                    W, H, W_orig, H_orig, test_imgs, img1, img2, out_path,
                    color, f"of_{name}", is_center, flow_fwd, out_path_flo,
                    add_optical_flow, mask)


def debug_calc_grid(x_array, y_array, q_start_x, q_start_y, k_start_x, k_start_y,
                    q_bin_width, q_bin_height, k_bin_width, k_bin_height, q_grids,
                    k_grids, q_x, q_y, k_x, k_y, test_imgs, img1, img2,
                    out_path, out_path_center, color, W_orig, H_orig,
                    center_q_x=None, center_q_y=None, center_k_x=None, center_k_y=None,
                    flow_fwd=None, out_path_flo="", out_path_center_flo="",
                    add_optical_flow=None, mask=None):
    H, W = x_array.shape[-2:]
    is_calc_flow = add_optical_flow is not None
    if mask is not None and isinstance(mask, list):
        mask_fwd, _ = mask

    # debug
    main_debug_calc_grid(q_start_x, q_start_y, k_start_x, k_start_y,
                         q_bin_width, q_bin_height, k_bin_width, k_bin_height, q_grids,
                         k_grids, q_x, q_y, k_x, k_y, test_imgs, img1, img2,
                         out_path_center, color, W_orig, H_orig,
                         center_q_x, center_q_y, center_k_x, center_k_y,
                         flow_fwd, out_path_center_flo, add_optical_flow,
                         mask, name="plot_point", is_center=True)
    # debug no center
    out_grids = calc_grid_no_center(x_array, y_array, q_bin_width, q_bin_height,
                                    k_bin_width, k_bin_height, q_start_x, q_start_y,
                                    k_start_x, k_start_y, W_orig, H_orig)
    q_x_n, q_y_n, k_x_n, k_y_n = out_grids
    if is_calc_flow:
        size = (H_orig, W_orig)
        center_q_x_n, center_q_y_n, mask_fwd = add_optical_flow(flow_fwd, q_x_n, q_y_n,
                                                                size, mask_fwd)
        center_k_x_n, center_k_y_n = k_x_n.clone(), k_y_n.clone()

    main_debug_calc_grid(q_start_x, q_start_y, k_start_x, k_start_y,
                         q_bin_width, q_bin_height, k_bin_width, k_bin_height, q_grids,
                         k_grids, q_x_n, q_y_n, k_x_n, k_y_n, test_imgs, img1, img2,
                         out_path, color, W_orig, H_orig,
                         center_q_x_n, center_q_y_n, center_k_x_n, center_k_y_n,
                         flow_fwd, out_path_flo, add_optical_flow,
                         mask, name="plot_point", is_center=False)
