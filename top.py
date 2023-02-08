#!/usr/bin/env python3

# Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved

import bisect
import curses
import json
import math
import os
import select
import signal
import threading
import time
import queue


class Const:
    SCREEN_MARGIN = 1
    MAX_MODEL_NAME_LEN = 50
    METRIC_GROUP_NAMES = ('neuroncore_counters', 'neuron_runtime_vcpu_usage', 'memory_used')
    SYSTEM_METRIC_GROUP_NAMES = ('memory_info', 'vcpu_usage')


class Utils:
    """ General use static functions
    """
    @staticmethod
    def human_readable_byte_size(size, right_align=True, fixed_width=True):
        suffix = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB')
        remaining = float(size)
        format_spec = '%.1f%s'
        if fixed_width:
            format_spec = '%6.1f%s' if right_align else '%-6.1f%s'
        for unit in suffix:
            if remaining < 1024:
                return format_spec % (remaining, unit)
            remaining = remaining / 1024
        return 'unknown'

    @staticmethod
    def format_percentage(perc):
        return '  100%' if perc > 99.995 else '%5.2f%%' % (perc,)

    @staticmethod
    def prop_getter(arr, idx):
        return arr[idx] if idx < len(arr) else arr[0]


class ScreenManager:
    """ Class managing the screen, handling color management and providing support for common draw operations
        (such as drawing a utilization bar).
    """
    def __init__(self):
        self.surface = curses.initscr()
        self.surface.nodelay(True)
        self.surface.keypad(True)
        self.surface_size = self.surface.getmaxyx()
        self.available_color_attr = {}
        curses.start_color()
        curses.curs_set(0)
        curses.noecho()
        curses.nonl()
        try:
            curses.cbreak()
        except:
            pass
        self.update_screen_size()

    def __del__(self):
        curses.endwin()

    @staticmethod
    def dim(color_attr):
        return color_attr | curses.A_DIM

    @staticmethod
    def bright(color_attr):
        return color_attr | curses.A_BOLD

    @staticmethod
    def make_color_key(color, bg_color):
        return (color << 8) | bg_color

    def safe_addstr(self, row, col, text, attrs=None):
        if attrs is None:
            attrs = self.get_default_color_attr()
        try:
            self.surface.addstr(row, col, text, attrs)
        except:
            pass

    def get_color_attr(self, color, bg_color=curses.COLOR_BLACK):
        color_key = ScreenManager.make_color_key(color, bg_color)
        if color_key in self.available_color_attr:
            return curses.color_pair(self.available_color_attr[color_key])
        nb_entries = len(self.available_color_attr)
        if nb_entries >= curses.COLORS:
            return self.get_default_color_attr()
        new_index = nb_entries + 1
        curses.init_pair(new_index, color, bg_color)
        self.available_color_attr[color_key] = new_index
        return curses.color_pair(new_index)

    def get_default_color_attr(self):
        return curses.color_pair(0)

    def update_screen_size(self):
        surface_size = self.surface.getmaxyx()
        just_resized = self.surface_size != surface_size
        self.surface_size = surface_size
        return just_resized

    def refresh(self):
        self.surface.refresh()

    def get_width(self):
        return self.surface_size[1]

    def get_height(self):
        return self.surface_size[0]

    def get_size(self):
        return self.surface_size[1], self.surface_size[0]

    def draw_ratio_bar(self, pos_x, pos_y, title_text, value_text, width, ratios, clrs, highlight=False):
        """ Draws a bar which has several sections, each displayed in a different color

        The format of the bar is title_text ||||||||||||value_text and has a length of width (including the texts),
        the color of each section is provided by the clrs list and the length of each section is provided by ratios.
        Parameters
        ----------
        pos_x : int
            x position where the bar will be drawn
        pos_y : int
            y position where the bar will be drawn
        title_text: string
            text displayed on the left side of the bar
        value_text: string
            text displayed on the right side of the bar
        width: int
            total length of the bar (including the texts, the actual length of the bar will be adjusted based on
            this parameter and the length of title_text and value_text)
        ratios: list
            a list of values between 0 and 1 representing the length of each section
        clrs: list
            a list of colors to draw each section, has to have the same number of elements as ratios
        highlight: boolean
            if true, the bar and the texts will have a different background color (default False)
        """
        total_text_width = len(value_text)
        title_text_width = len(title_text)
        if title_text_width > 0:
            total_text_width += title_text_width + 1
        bar_width = width - total_text_width
        if bar_width <= 0:
            return pos_x
        self.safe_addstr(pos_y, pos_x, ' ' * width, self.get_default_color_attr())
        if title_text_width > 0:
            self.safe_addstr(pos_y, pos_x, title_text + ' ', self.get_default_color_attr())
            pos_x += title_text_width + 1
        prev_remainder = 0.0
        remaining_width = bar_width
        for idx, ratio in enumerate(ratios):
            used_width = ratio * bar_width
            used_width -= prev_remainder
            if used_width >= 0.0:
                prev_remainder = 0.0
            else:
                prev_remainder = -used_width
                used_width = 0.0
            remainder = used_width - math.floor(used_width)
            used_width = int(used_width)
            if used_width == 0:
                continue
            if remainder > 0:
                prev_remainder = (1.0 - remainder)
            remainder_width = 0
            if remainder > 0.25:
                remainder_width = 1
            else:
                prev_remainder = 0
            color = self.get_color_attr(clrs[idx])
            if highlight:
                color = self.get_color_attr(clrs[idx], curses.COLOR_BLUE)
            for part in ((used_width, ScreenManager.bright(color)),
                         (remainder_width, ScreenManager.dim(color))):
                if part[0] == 0:
                    continue
                self.safe_addstr(pos_y, pos_x, '|' * part[0], part[1])
                pos_x += part[0]
                remaining_width -= part[0]
        fill_color = self.get_default_color_attr()
        if highlight:
            fill_color = self.get_color_attr(curses.COLOR_WHITE, curses.COLOR_BLUE)
        if remaining_width > 0:
            self.safe_addstr(pos_y, pos_x, '|' * remaining_width, ScreenManager.dim(fill_color))
            pos_x += remaining_width
        self.safe_addstr(pos_y, pos_x, value_text, self.get_default_color_attr())
        return pos_x + len(value_text)

    def clear_section(self, pos_y, height):
        scr_w, scr_h = self.get_size()
        for crt_y in range(pos_y, pos_y + height):
            if crt_y < 0 or crt_y >= scr_h:
                continue
            self.safe_addstr(crt_y, 0, ' ' * scr_w, self.get_default_color_attr())

    def clear_screen(self):
        self.surface.clear()
        self.refresh()


class Widget:
    """ Base class for all screen widgets.
    """
    REFRESH_MODE_NONE = 0
    REFRESH_MODE_DATA = 1
    REFRESH_MODE_FULL = (1 << 2)

    def __init__(self, screen, pos_y, height):
        self.screen = screen
        self.refresh_mode = Widget.REFRESH_MODE_FULL
        self.pos_y = pos_y
        self.height = height
        self.visible = True

    def check_bounds(self):
        height = self.screen.get_height()
        return self.pos_y >= 0 and self.pos_y < height

    def refresh(self):
        pass

    def recalculate_layout(self):
        pass

    def clear(self):
        self.screen.clear_section(self.pos_y, self.height)

    def set_pos_y(self, new_y):
        if self.pos_y != new_y:
            prev_pos_y = self.pos_y
            self.pos_y = new_y
            self.on_pos_y_changed(prev_pos_y)

    def set_height(self, new_height):
        if self.height != new_height:
            prev_height = self.height
            self.height = new_height
            self.on_height_changed(prev_height)

    def set_visible(self, visible):
        if self.visible != visible:
            prev_visible = self.visible
            self.visible = visible
            self.on_visible_changed(prev_visible)

    def on_pos_y_changed(self, prev_y):
        self.screen.clear_section(prev_y, self.height)
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_height_changed(self, prev_height):
        self.recalculate_layout()
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_visible_changed(self, prev_visible):
        if self.visible is False:
            self.screen.clear_section(self.pos_y, self.height)
        else:
            self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_data_updated(self):
        self.refresh_mode |= Widget.REFRESH_MODE_DATA

    def on_screen_resize(self):
        self.recalculate_layout()
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def refresh_requested(self):
        return self.refresh_mode != 0

    def on_post_refresh(self):
        self.refresh_mode = 0

    def is_refresh_mode_requested(self, mode):
        return (self.refresh_mode & mode) != 0

    def request_refresh_mode(self, mode):
        self.refresh_mode |= mode

    def on_key(self, key):
        pass


class LabelWidget(Widget):
    """ Simple widget that displays one or more labels on a single line.
    """
    ALIGN_LEFT = 0
    ALIGN_RIGHT = 1
    ALIGN_CENTER = 2

    def __init__(self, screen, pos_y, margins, texts, aligns, colors, bg_color):
        Widget.__init__(self, screen, pos_y, 1)
        self.colors = colors
        self.margins = margins
        self.bg_color = bg_color
        self.set_texts(texts, aligns)

    def set_texts(self, texts, aligns):
        self.texts = texts
        self.start_x = 0
        self.width = 0
        self.aligns = aligns
        self.text_pos_x = [0] * len(texts)
        self.text_visible = [False] * len(texts)
        self.recalculate_layout()

    def set_colors(self, colors, bg_color):
        self.colors = colors
        self.bg_color = bg_color
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def recalculate_layout(self):
        width = self.screen.get_width()
        col_count = len(self.texts)
        if col_count == 0:
            return
        text_area_width = width - sum(self.margins) - Const.SCREEN_MARGIN * 2
        base_col_width = text_area_width // col_count
        col_width_remaining = text_area_width % col_count
        if base_col_width <= 0:
            return
        self.start_x = Const.SCREEN_MARGIN + self.margins[0]
        self.width = text_area_width
        crt_x = self.start_x
        for idx, label in enumerate(self.texts):
            text_width = len(label)
            col_width = base_col_width
            if col_width_remaining > 0:
                col_width += 1
                col_width_remaining -= 1
            self.text_visible[idx] = text_width <= col_width
            pos_x = crt_x
            align = Utils.prop_getter(self.aligns, idx)
            if align == LabelWidget.ALIGN_RIGHT:
                pos_x = crt_x + col_width - text_width
            elif align == LabelWidget.ALIGN_CENTER:
                pos_x = crt_x + (col_width - text_width) // 2
            self.text_pos_x[idx] = pos_x
            crt_x += col_width
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def refresh(self):
        if not self.check_bounds():
            return
        if not self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
            return
        color_attr = self.screen.get_color_attr(Utils.prop_getter(self.colors, 0), self.bg_color)
        self.screen.safe_addstr(self.pos_y, self.start_x, ' ' * self.width, color_attr)
        for idx, label in enumerate(self.texts):
            if not self.text_visible[idx]:
                continue
            color_attr = self.screen.get_color_attr(Utils.prop_getter(self.colors, idx), self.bg_color)
            self.screen.safe_addstr(self.pos_y, self.text_pos_x[idx], label, color_attr)


class NCUsageWidget(Widget):
    """ NeuronCore utilization widget, shows 4 utilization bars representing the NeuronCore utilization for a
        Neuron device.
    """
    def __init__(self, screen, pos_y, nd_index, nc_per_nd):
        Widget.__init__(self, screen, pos_y, 1)
        self.nd_index = nd_index
        self.nc_per_nd = nc_per_nd
        self.utilization = [0.0] * nc_per_nd
        self.highlight = [False] * nc_per_nd
        self.bar_updated = [True] * nc_per_nd
        self.bar_widths = [0] * nc_per_nd
        self.bar_pos_x = [0] * nc_per_nd
        self.bar_visible = [False] * nc_per_nd
        self.recalculate_layout()

    def update_data(self, utilization):
        data_updated = False
        for idx, util in enumerate(utilization):
            if self.utilization[idx] != util:
                self.bar_updated[idx] = True
                self.utilization[idx] = util
                data_updated = True
        if data_updated:
            self.on_data_updated()

    def set_highlight(self, index, show=True):
        if index == -1:
            index = 0
            end_idx = self.nc_per_nd
        else:
            end_idx = index + 1
        highlight_updated = False
        for idx in range(index, end_idx):
            if show != self.highlight[idx]:
                self.highlight[idx] = show
                self.bar_updated[idx] = True
                highlight_updated = True
        if highlight_updated:
            self.on_data_updated()

    def recalculate_layout(self):
        width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2)
        col_width = width - Const.SCREEN_MARGIN - self.nc_per_nd + 1
        remaining_width = col_width % self.nc_per_nd
        col_width //= self.nc_per_nd
        crt_x = Const.SCREEN_MARGIN
        for idx, _ in enumerate(self.utilization):
            adj_col_width = col_width
            if remaining_width > 0:
                adj_col_width += 1
                remaining_width -= 1
            self.bar_pos_x[idx] = crt_x
            self.bar_widths[idx] = col_width
            crt_x += col_width + 1

    def refresh(self):
        """ NDxx ########[99.99%] ########[99.99%] ########[99.99%] ########[99.99%]
        """
        if not self.check_bounds():
            return
        nd_text = 'ND%-2d' % (self.nd_index,)
        for idx, util in enumerate(self.utilization):
            if self.bar_updated[idx] or self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
                percent_text = '[%s]' % (Utils.format_percentage(util))
                self.screen.draw_ratio_bar(self.bar_pos_x[idx], self.pos_y, nd_text, percent_text, self.bar_widths[idx],
                                           (util / 100.0, ), (curses.COLOR_GREEN, ), self.highlight[idx])
                self.bar_updated[idx] = False
            nd_text = ''


class RuntimeMemoryUsageWidget(Widget):
    """ Runtime memory utilization widget, shows an utilization bar indicating the host memory usage, and displays
        the Neuron device memory usage.
    """
    SPACING = 4
    def __init__(self, screen, pos_y):
        Widget.__init__(self, screen, pos_y, 1)
        self.bytes_host = 0
        self.bytes_host_total = 0
        self.bytes_device = 0
        self.recalculate_layout()

    def update_data(self, bytes_host, bytes_host_total, bytes_device):
        if self.bytes_host == bytes_host and self.bytes_host_total == bytes_host_total and \
           self.bytes_device == bytes_device:
            return
        self.bytes_host = bytes_host if bytes_host != -1 else self.bytes_host
        self.bytes_host_total = bytes_host_total if bytes_host_total != -1 else self.bytes_host_total
        self.bytes_device = bytes_device
        self.on_data_updated()

    def recalculate_layout(self):
        width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2) - RuntimeMemoryUsageWidget.SPACING
        self.col_width = width // 2
        self.bar_pos_x = Const.SCREEN_MARGIN
        self.text_pos_x = (self.screen.get_width() + RuntimeMemoryUsageWidget.SPACING) // 2

    def refresh(self):
        """ Runtime memory host ########[xxxx.xGB/xxxx.xGB]  Runtime memory device: xxxx.xxGB
        """
        if not self.check_bounds():
            return
        self.clear()
        title_1 = 'Runtime Memory Host'
        title_2 = 'Runtime Memory Device'
        mem_usage = 0.0
        if self.bytes_host_total != 0:
            mem_usage = self.bytes_host / self.bytes_host_total
        mem_usage = min(1.0, mem_usage)
        mem_usage_text = '[%s/%s]' % (Utils.human_readable_byte_size(self.bytes_host),
                                      Utils.human_readable_byte_size(self.bytes_host_total))
        self.screen.draw_ratio_bar(self.bar_pos_x, self.pos_y, title_1, mem_usage_text, self.col_width, (mem_usage, ),
                                   (curses.COLOR_CYAN, ))
        mem_usage_device_text = '%s %s' % (title_2, Utils.human_readable_byte_size(self.bytes_device))
        if len(mem_usage_device_text) < self.col_width:
            self.screen.safe_addstr(self.pos_y, self.text_pos_x,
                                    mem_usage_device_text, self.screen.get_default_color_attr())


class VCPUUsageWidget(Widget):
    """ vCPU utilization widget, shows an utilization bar indicating the total system average vCPU usage and one
        displaying the percent of vCPU used exclusively by the Neuron Runtime.
    """
    SPACING = 4
    def __init__(self, screen, pos_y):
        Widget.__init__(self, screen, pos_y, 1)
        self.system_cpu_usage = {
            'user': 0,
            'system': 0,
            'needs_refresh': True
        }
        self.neuron_runtime_cpu_usage = {
            'user': 0,
            'system': 0,
            'needs_refresh': True
        }
        self.recalculate_layout()

    def recalculate_layout(self):
        width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2) - VCPUUsageWidget.SPACING
        self.col_width = width // 2
        self.bar_pos_x = (Const.SCREEN_MARGIN, (self.screen.get_width() + VCPUUsageWidget.SPACING) // 2)

    def update_data(self, system_cpu_usage, neuron_runtime_cpu_usage):
        found_updates = False
        for entry in ((self.system_cpu_usage, system_cpu_usage),
                      (self.neuron_runtime_cpu_usage, neuron_runtime_cpu_usage)):
            if entry[1] is None:
                continue
            for field in entry[1]:
                if entry[0][field] != entry[1][field]:
                    entry[0][field] = entry[1][field]
                    entry[0]['needs_refresh'] = True
                    found_updates = True
        if found_updates:
            self.on_data_updated()

    def refresh(self):
        """ System vCPU Usage ########[99.99%,99.99%]    Runtime vCPU Usage ########[99.99%,99.99%]
        """
        if not self.check_bounds():
            return
        full_refresh = self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL)
        if full_refresh:
            self.clear()

        title_1 = 'System vCPU Usage'
        title_2 = 'Runtime vCPU Usage'

        for idx, entry in enumerate(((self.system_cpu_usage, title_1),
                                     (self.neuron_runtime_cpu_usage, title_2))):
            if entry[0]['needs_refresh'] or full_refresh:
                percent_text = '[%s,%s]' % (Utils.format_percentage(entry[0]['user']),
                                            Utils.format_percentage(entry[0]['system']))
                self.screen.draw_ratio_bar(self.bar_pos_x[idx], self.pos_y, entry[1], percent_text, self.col_width,
                                           (entry[0]['user'] / 100.0, entry[0]['system'] / 100.0),
                                           (curses.COLOR_GREEN, curses.COLOR_RED))
                entry[0]['needs_refresh'] = False


class ModelMemoryUsageListWidget(Widget):
    """ Model tree view widget, displays the currently loaded models in a tree-like structure which responds
        to key pressed (up/down/left/right/x). It also indicates the amount of host memory and Neuron Device
        memory used by each.
    """
    ITEM_TYPE_ND = 0
    ITEM_TYPE_NC = 1
    ITEM_TYPE_MD = 2

    SELECT_NEXT = 0
    SELECT_PREV = 1

    class UsageListItem:
        """ Single item of a ModelMemoryUsageListWidget
        """
        def __init__(self, parent, identifier, text, item_type):
            self.selected = False
            self.expanded = False
            self.is_active = False
            self.needs_refresh = True
            self.dimmed = False
            self.text = text
            self.host_mem = 0
            self.device_mem = 0
            self.parent = parent
            self.identifier = identifier
            self.item_type = item_type
            self.index = 0
            self.screen_pos_y = None
            self.children = []

        def set_data(self, host_mem, device_mem, is_dimmed):
            if self.host_mem != host_mem or self.device_mem != device_mem or self.dimmed != is_dimmed:
                self.host_mem = host_mem
                self.device_mem = device_mem
                self.dimmed = is_dimmed
                self.needs_refresh = True
            self.is_active = True

        def refresh(self, screen, pos_x, pos_y, base_pos_y, values_x, max_x, max_y, full_refresh):
            """
            [+] ND0                                      1221MB              16MB
                [+] NC0                                   100MB               8MB
                      myModel                 11223344     50MB               4MB
            """
            needs_refresh = self.needs_refresh or full_refresh
            self.needs_refresh = False
            if needs_refresh and pos_y >= base_pos_y and pos_y <= max_y:
                screen.clear_section(pos_y, 1)
                color_attr = screen.get_default_color_attr()
                if self.dimmed:
                    color_attr = ScreenManager.dim(color_attr)
                text = ''
                if self.children:
                    if self.expanded:
                        text = '[-] '
                    else:
                        text = '[+] '
                text += self.text
                crt_max_x = pos_x + len(text) - 1
                if crt_max_x < values_x[0]:
                    screen.safe_addstr(pos_y, pos_x, text, color_attr)
                for idx, value in enumerate((self.identifier, self.host_mem, self.device_mem)):
                    if idx == 0 and self.item_type is not ModelMemoryUsageListWidget.ITEM_TYPE_MD:
                        continue
                    added_offset = 0
                    if idx == 0:
                        text = '%d' % (self.identifier, )
                    else:
                        text = Utils.human_readable_byte_size(value, False, False)
                        if self.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_NC:
                            added_offset = 1
                        elif self.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_MD:
                            added_offset = 2
                    text_x = values_x[idx] + added_offset
                    if text_x + len(text) >= max_x:
                        break
                    screen.safe_addstr(pos_y, text_x, text, color_attr)
                if self.selected:
                    color_attr = screen.get_color_attr(curses.COLOR_YELLOW, curses.COLOR_RED)
                    screen.surface.chgat(pos_y, pos_x, max_x - pos_x - 1, color_attr)
            self.screen_pos_y = pos_y
            pos_y += 1
            if self.expanded:
                for item in self.children:
                    pos_y = item.refresh(screen, pos_x + 4, pos_y, base_pos_y, values_x, max_x, max_y, full_refresh)
            return pos_y

        def __lt__(self, other):
            return self.identifier < other.identifier

    def __init__(self, screen, pos_y, bottom_margin):
        Widget.__init__(self, screen, pos_y, screen.get_height() - bottom_margin - pos_y)
        self.children = []
        self.bottom_margin = bottom_margin
        self.selected = None
        self.scroll_amount = 0
        self.base_y = None
        self.selection_changed_handler = None
        self.fully_expanded = False
        self.recalculate_layout()

    def set_selection_changed_handler(self, handler):
        self.selection_changed_handler = handler

    def _parse_tree(self, root, func):
        func(root)
        for child in root.children:
            self._parse_tree(child, func)

    def _pre_update(self):

        def pre_updater(node):
            if node is not self:
                node.is_active = False

        self._parse_tree(self, pre_updater)

    def _post_update(self):
        to_delete = {}
        selected_deleted = False

        def post_updater(node):
            nonlocal selected_deleted
            for idx, child in enumerate(node.children):
                if child.is_active is True:
                    continue
                if node not in to_delete:
                    to_delete[node] = set()
                to_delete[node].add(idx)
                if self.selected is child:
                    selected_deleted = True

        self._parse_tree(self, post_updater)
        if to_delete:
            for deleted, deleted_indices in to_delete.items():
                deleted.children = [c for i, c in enumerate(deleted.children) if i not in deleted_indices]
                for idx, child in enumerate(deleted.children):
                    child.index = idx
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        if not self.children:
            self._select_new_item(None)
        elif selected_deleted or self.selected is None:
            self._select_new_item(self.children[0])
        else:
            self._make_selected_item_visible()

    def add_or_get_child(self, parent, identifier, text, item_type):
        temp = ModelMemoryUsageListWidget.UsageListItem(parent, identifier, text, item_type)
        idx = 0
        if parent.children:
            idx = bisect.bisect_left(parent.children, temp)
            if idx < len(parent.children) and parent.children[idx].identifier == identifier:
                return parent.children[idx]
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        parent.children.insert(idx, temp)
        for upd_idx in range(idx, len(parent.children)):
            parent.children[upd_idx].index = upd_idx
        return parent.children[idx]

    def recalculate_layout(self):
        width, height = self.screen.get_size()
        self.max_x = width - Const.SCREEN_MARGIN
        self.max_y = height - self.bottom_margin - 1
        start_x = width // 3
        col_width = width // 6
        self.values_pos_x = (max(start_x + col_width, 100), max(start_x + col_width * 2, 120),
                             max(start_x + col_width * 3, 140))
        new_base_y = self.pos_y + 1
        if self.base_y != new_base_y:
            self.base_y = new_base_y
            self._refresh_items_screen_y_pos()

    def refresh(self):
        full_refresh = self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL)
        if full_refresh:
            self.screen.clear_section(self.pos_y, self.max_y - self.pos_y + 1)
        if self.children:
            for idx, text in enumerate(('Model ID', 'Host Memory', 'Device Memory')):
                if self.values_pos_x[idx] + len(text) >= self.max_x:
                    break
                self.screen.safe_addstr(self.pos_y, self.values_pos_x[idx], text)
        crt_y = self.base_y - self.scroll_amount
        for item in self.children:
            crt_y = item.refresh(self.screen, 2, crt_y, self.base_y, self.values_pos_x,
                                 self.max_x, self.max_y, full_refresh)

    def on_key(self, key):
        if key == curses.KEY_UP:
            self._select_prev()
        elif key == curses.KEY_DOWN:
            self._select_next()
        if key == curses.KEY_LEFT:
            self._collapse_and_select()
        elif key == curses.KEY_RIGHT:
            self._expand_and_select()
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            self._expand_collapse_selected()
        elif key in [ord('x'), ord('X')]:
            self._toggle_expand_all()

    def on_pos_y_changed(self, prev_y):
        self.height = self.screen.get_height() - self.bottom_margin - self.pos_y
        Widget.on_pos_y_changed(self, prev_y)
        self.recalculate_layout()

    def _get_last_expanded_item(self, root):
        while True:
            if root.expanded is False or not root.children:
                return root
            root = root.children[-1]

    def _get_next_available_item(self, current):
        while True:
            if current.index < len(current.parent.children) - 1:
                return current.parent.children[current.index + 1]
            if current.parent == self:
                return self.children[0]
            current = current.parent

    def _get_next_item(self, direction):
        crt_index = self.selected.index
        if direction == ModelMemoryUsageListWidget.SELECT_PREV:
            if crt_index > 0:
                return self._get_last_expanded_item(self.selected.parent.children[crt_index - 1])
            if self.selected.parent is not self:
                return self.selected.parent
            return self._get_last_expanded_item(self.children[-1])
        if self.selected.expanded and self.selected.children:
            return self.selected.children[0]
        return self._get_next_available_item(self.selected)

    def _make_selected_item_visible(self):
        if self.selected is not None:
            if self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
                self._refresh_items_screen_y_pos()
            bound = (self.base_y + 1, self.screen.get_height() - self.bottom_margin - 1)
            if self.selected.screen_pos_y < bound[0]:
                self.scroll_amount = max(0, self.scroll_amount - (bound[0] - self.selected.screen_pos_y))
                self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            elif self.selected.screen_pos_y > bound[1]:
                self.scroll_amount += (self.selected.screen_pos_y - bound[1])
                self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def _select_new_item(self, new_item):
        old_item = self.selected
        if old_item is not new_item:
            if old_item is not None:
                old_item.selected = False
                old_item.needs_refresh = True
            if new_item is not None:
                new_item.selected = True
                new_item.needs_refresh = True
            self.selected = new_item
            self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
            self._make_selected_item_visible()
            if self.selection_changed_handler is not None:
                self.selection_changed_handler.on_selection_changed(old_item, new_item)

    def _select_prev(self):
        if self.selected is None:
            return
        self._select_new_item(self._get_next_item(ModelMemoryUsageListWidget.SELECT_PREV))

    def _select_next(self):
        if self.selected is None:
            return
        self._select_new_item(self._get_next_item(ModelMemoryUsageListWidget.SELECT_NEXT))

    def _collapse_and_select(self):
        if self.selected is None:
            return
        if self.selected.expanded is True:
            self.selected.expanded = False
            for child in self.selected.children:
                child.expanded = False
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            return
        parent = self.selected.parent
        if parent is self:
            self._select_new_item(self.children[0])
        else:
            self._select_new_item(parent)

    def _expand_and_select(self):
        if self.selected is None:
            return
        if self.selected.expanded is False:
            self.selected.expanded = True
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            return
        if self.selected.children:
            self._select_new_item(self.selected.children[0])

    def _expand_collapse_selected(self):
        if self.selected is None:
            return
        if self.selected.children:
            self.selected.expanded = not self.selected.expanded
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def get_top_parent(self, node):
        while node.parent is not self:
            node = node.parent
        return node

    def _refresh_items_screen_y_pos(self):
        crt_y = self.base_y - self.scroll_amount

        def y_updater(node):
            nonlocal crt_y
            if node is self:
                return
            node.screen_pos_y = crt_y
            if node.parent is self or node.parent.expanded is True:
                crt_y += 1

        self._parse_tree(self, y_updater)

    def _expand_collapse_all(self, expanded=True):
        updated_exp = False

        def expand_setter(node):
            nonlocal updated_exp
            if node is self:
                return
            if node.expanded is not expanded and len(node.children) > 0:
                node.expanded = expanded
                updated_exp = True

        self._parse_tree(self, expand_setter)
        if updated_exp is False:
            return
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        if expanded is False and self.selected is not None:
            new_selection = self.get_top_parent(self.selected)
            self._select_new_item(new_selection)
        else:
            self._make_selected_item_visible()

    def _toggle_expand_all(self):
        new_val = not self.fully_expanded
        self._expand_collapse_all(new_val)
        self.fully_expanded = new_val

    def update_data(self, list_data):
        self._pre_update()
        nd_indices = sorted(list_data.keys())
        for nd_idx in nd_indices:
            nd_obj = list_data[nd_idx]
            nd_item = self.add_or_get_child(self, nd_idx, 'ND%2d' % (nd_idx,), self.ITEM_TYPE_ND)
            nd_item.set_data(nd_obj['host'], nd_obj['neuron_device'], False)
            nc_indices = sorted(nd_obj['children'].keys())
            for nc_idx in nc_indices:
                nc_obj = nd_obj['children'][nc_idx]
                nc_item = self.add_or_get_child(nd_item, nc_idx, 'NC%d' % (nc_idx,), self.ITEM_TYPE_NC)
                nc_item.set_data(nc_obj['host'], nc_obj['neuron_device'], False)
                for model, model_obj in nc_obj['children'].items():
                    model_name = model_obj['name']
                    model_name = model_name[len(model_name) - Const.MAX_MODEL_NAME_LEN:]
                    model_item = self.add_or_get_child(nc_item, model, model_name, self.ITEM_TYPE_MD)
                    model_item.set_data(model_obj['host'], model_obj['neuron_device'], not model_obj['is_running'])
                    if model_item.needs_refresh:
                        self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
        self._post_update()


class NeuronMonitorTop:
    """ Main application class, manages the widgets, the reader thread which gets data from neuron-monitor
        and the state of the application.
    """
    STATE_INVALID = ''
    STATE_LOADING = 'state_loading'
    STATE_MAIN = 'state_main'

    UPDATE_TYPE_DATA = 0
    UPDATE_TYPE_KEYS = 1
    UPDATE_TYPE_SCR_RESIZE = 2
    UPDATE_TYPE_CHANGE_STATE = 3
    UPDATE_TYPE_QUIT = 4

    RUNTIME_TAB_COUNT = 5
    MAX_KEY_EVENTS_IN_QUEUE = 2

    def __init__(self):
        self.json_stdin = os.fdopen(os.dup(0))
        os.dup2(os.open('/dev/tty', os.O_RDONLY), 0)

        self.running = False
        self.screen = ScreenManager()
        self.screen_resized = True
        self.widgets = {}
        self.label_widgets = {}
        self.key_listeners = set()
        self.available_runtimes = {}
        self.available_pids = []
        self.system_data = None
        self.current_runtime_pid = None
        self.state = None
        self.groups_next_y = 1
        self.monitor_ready = False
        self.refresh_runtime_tab = False
        self.runtime_tab_offset = 0
        self.neuron_device_count = 0
        self.neuroncore_count = 0
        self.rebuild_ui = False
        self.reset_ui = False
        self.update_queue = queue.Queue()
        self.data_update_thread = None
        self.ui_update_thread = None
        self.key_increment_lock = threading.Lock()
        self.key_events_in_queue = 0
        self.screen_update_lock = threading.Lock()
        self._change_state(NeuronMonitorTop.STATE_LOADING)

    def run(self):
        self.running = True

        def reader_thread(monitor):
            poller = select.poll()
            poller.register(monitor.json_stdin, select.POLLIN)
            while monitor.running:
                if not poller.poll(1):
                    time.sleep(0.1)
                    continue
                line = monitor.json_stdin.readline()
                if not line:
                    continue
                monitor_data = json.loads(line)
                monitor.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_DATA, monitor_data))

        def ui_update_thread(monitor):
            while monitor.running:
                pressed = -1
                while True:
                    with monitor.screen_update_lock:
                        crt_key = monitor.screen.surface.getch()
                    if crt_key == -1:
                        break
                    pressed = crt_key
                if pressed != -1:
                    monitor._add_key_event(pressed)
                screen_resized = monitor.screen.update_screen_size()
                if screen_resized:
                    monitor.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_SCR_RESIZE, None))
                time.sleep(0.1)

        self.data_update_thread = threading.Thread(None, reader_thread, args=(self, ))
        self.data_update_thread.start()
        self.ui_update_thread = threading.Thread(None, ui_update_thread, args=(self, ))
        self.ui_update_thread.start()
        signal.signal(signal.SIGINT, lambda signal, frame: self._stop())
        self._internal_loop()

    def _add_key_event(self, pressed):
        with self.key_increment_lock:
            if self.key_events_in_queue >= NeuronMonitorTop.MAX_KEY_EVENTS_IN_QUEUE:
                return
            self.key_events_in_queue += 1
            self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_KEYS, pressed))

    def _on_processed_key_event(self):
        with self.key_increment_lock:
            if self.key_events_in_queue > 0:
                self.key_events_in_queue -= 1

    def _call_widget_metric_group_func(self, prefix, create_widgets, runtime_data=None, system_data=None):
        for metric_group_name in Const.METRIC_GROUP_NAMES:
            if runtime_data and metric_group_name not in runtime_data:
                continue
            metric_group_data = runtime_data[metric_group_name] if runtime_data else None
            handler_name = prefix + metric_group_name
            handler_method = getattr(self, handler_name, None)
            if handler_method is None:
                continue
            if create_widgets and metric_group_name not in self.widgets:
                self.widgets[metric_group_name] = {}
            widget_group = self.widgets[metric_group_name]
            handler_method(widget_group, metric_group_data, system_data)

    def _build_ui(self):
        self._call_widget_metric_group_func('_build_ui_', True)

    def _build_ui_neuroncore_counters(self, widget_group, *_):
        crt_y = self.groups_next_y

        self._set_label('neuroncore_info_title', crt_y, (0, 0), ('NeuronCore Utilization', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        self._set_label('neuroncore_info_header', crt_y + 1, (4, 0),
                        ['NC%d' % (nc, ) for nc in range(self.neuroncore_count)],
                        (LabelWidget.ALIGN_CENTER, ), (curses.COLOR_WHITE, ), curses.COLOR_BLACK)
        crt_y += 2
        for crt_idx in range(0, self.neuron_device_count):
            if crt_idx not in widget_group:
                widget_group[crt_idx] = NCUsageWidget(self.screen, crt_y, crt_idx, self.neuroncore_count)
            crt_y += 1
        self.groups_next_y = crt_y + 1

    def _build_ui_memory_used(self, widget_group, *_):
        crt_y = self.groups_next_y

        key_name = 'neuron_runtime_used_bytes'
        if key_name not in widget_group:
            widget_group[key_name] = RuntimeMemoryUsageWidget(self.screen, crt_y)
        else:
            widget_group[key_name].set_pos_y(crt_y)
        crt_y += 2

        self._set_label('models_info_title', crt_y, (0, 0), ('Loaded Models', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        crt_y += 1

        key_name = 'runtime_memory_models'
        model_list_widget = None
        if key_name not in widget_group:
            model_list_widget = ModelMemoryUsageListWidget(self.screen, crt_y, 2)
            model_list_widget.set_selection_changed_handler(self)
            widget_group[key_name] = model_list_widget
            self._register_key_listener(model_list_widget)
        else:
            model_list_widget = widget_group[key_name]
            model_list_widget.set_pos_y(crt_y)

    def _build_ui_neuron_runtime_vcpu_usage(self, widget_group, *_):
        crt_y = self.groups_next_y

        self._set_label('vcpu_mem_title', crt_y, (0, 0), ('vCPU and Memory Info', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        crt_y += 1

        key_name = 'neuron_runtime_vcpu_usage'
        if key_name not in widget_group:
            widget_group[key_name] = VCPUUsageWidget(self.screen, crt_y)
        else:
            widget_group[key_name].set_pos_y(crt_y)
        self.groups_next_y = crt_y + 1

    def _set_neuroncore_util_highlight(self, nc_widgets, selected_item, show=True):
        if selected_item.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_ND:
            nd_idx = selected_item.identifier
            if nd_idx in nc_widgets:
                nc_widgets[nd_idx].set_highlight(-1, show)
        else:
            if selected_item.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_MD:
                selected_item = selected_item.parent
            nc_idx = selected_item.identifier
            selected_item = selected_item.parent
            nd_idx = selected_item.identifier
            if nd_idx in nc_widgets:
                nc_widgets[nd_idx].set_highlight(nc_idx, show)

    def on_selection_changed(self, old_item, new_item):
        widgets = self.widgets['neuroncore_counters']
        if old_item is not None:
            self._set_neuroncore_util_highlight(widgets, old_item, False)
        if new_item is not None:
            self._set_neuroncore_util_highlight(widgets, new_item, True)

    def _change_state(self, new_state):
        if self.state is new_state:
            return
        self.state = new_state
        self.widgets = {}
        self.label_widgets = {}
        self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_CHANGE_STATE, None))
        if new_state == NeuronMonitorTop.STATE_MAIN:
            self.refresh_runtime_tab = True
            self.rebuild_ui = True
        self.screen.clear_screen()

    def _init_aggregated_neuroncore_counters(self, destination):
        destination['error'] = ''
        destination['neuroncores_in_use'] = {}

    def _aggregate_neuroncore_counters(self, destination, source):
        if source['error'] != '':
            return
        nc_dest = destination['neuroncores_in_use']
        nc_src = source['neuroncores_in_use']
        for nc_idx in nc_src:
            nc_util_src = nc_src[nc_idx]['neuroncore_utilization']
            if nc_idx in nc_dest:
                nc_dest[nc_idx]['neuroncore_utilization'] = min(
                    100, nc_dest[nc_idx]['neuroncore_utilization'] + nc_util_src)
            else:
                nc_dest[nc_idx] = {
                    'neuroncore_utilization': nc_util_src
                }

    def _init_aggregated_memory_used(self, destination):
        destination['error'] = ''
        destination['neuron_runtime_used_bytes'] = {
            'host': 0,
            'neuron_device': 0
        }
        destination['loaded_models'] = []

    def _aggregate_memory_used(self, destination, source):
        if source['error'] != '':
            return
        used_src = source['neuron_runtime_used_bytes']
        used_dest = destination['neuron_runtime_used_bytes']
        for metric, value in used_src.items():
            used_dest[metric] += value
        destination['loaded_models'] += source['loaded_models'][:]

    def _init_aggregated_neuron_runtime_vcpu_usage(self, destination):
        destination['error'] = ''
        destination['vcpu_usage'] = {
            'user': 0,
            'system': 0
        }

    def _aggregate_neuron_runtime_vcpu_usage(self, destination, source):
        if source['error'] != '':
            return
        vcpu_usage_src = source['vcpu_usage']
        vcpu_usage_dest = destination['vcpu_usage']
        for metric, value in vcpu_usage_src.items():
            vcpu_usage_dest[metric] += value

    def _aggregate_current_runtimes(self, current_runtimes):
        aggregated = {}
        aggregated['report'] = {}
        aggregated['error'] = ''
        aggregated['neuron_runtime_tag'] = 'all'
        report = aggregated['report']
        for metric in Const.METRIC_GROUP_NAMES:
            if metric not in report:
                report[metric] = {}
            handler_name = '_init_aggregated_' + metric
            handler_method = getattr(self, handler_name, None)
            handler_method(report[metric])

        for _, runtime_data in current_runtimes.items():
            current_report = runtime_data['report']
            for metric, metric_data in current_report.items():
                handler_name = '_aggregate_' + metric
                handler_method = getattr(self, handler_name, None)
                handler_method(report[metric], metric_data)
        return aggregated

    def _update_available_runtimes(self, monitor_data):
        current_runtimes = {}
        if monitor_data and monitor_data['neuron_runtime_data']:
            for item in monitor_data['neuron_runtime_data']:
                if item['error'] == '':
                    current_runtimes[item['pid']] = item
        current_runtimes[0] = self._aggregate_current_runtimes(current_runtimes)
        prev_runtime_count = len(self.available_runtimes)
        to_delete = []
        for pid, _ in self.available_runtimes.items():
            if pid not in current_runtimes:
                to_delete.append(pid)
                self.refresh_runtime_tab = True
                if self.current_runtime_pid == pid:
                    self.current_runtime_pid = None
        for pid in to_delete:
            del self.available_runtimes[pid]
        for pid, runtime in current_runtimes.items():
            if pid not in self.available_runtimes:
                self.refresh_runtime_tab = True
            self.available_runtimes[pid] = runtime
        if self.current_runtime_pid is None:
            self._select_new_pid(0)
        self.available_pids = sorted(self.available_runtimes.keys())
        self.reset_ui = prev_runtime_count > 0 and not self.available_pids

    def _update_data(self, monitor_data):
        self.system_data = None
        if monitor_data:
            self.system_data = monitor_data['system_data']
            neuron_hw_info = monitor_data['neuron_hardware_info']
            if neuron_hw_info['neuron_device_count'] > self.neuron_device_count:
                self.neuron_device_count = neuron_hw_info['neuron_device_count']
                self.neuroncore_count = neuron_hw_info['neuroncore_per_device_count']
                self.rebuild_ui = True
        self._update_available_runtimes(monitor_data)
        self.monitor_ready = True

    def _update_neuroncore_counters(self, widget_group, data, _):
        nd_data = {}
        for nc_idx, nc_data in data['neuroncores_in_use'].items():
            nc_idx = int(nc_idx)
            nd_idx = nc_idx // self.neuroncore_count
            nc_idx = nc_idx % self.neuroncore_count
            if nd_idx not in nd_data:
                nd_data[nd_idx] = [0.0] * self.neuroncore_count
            nd_data[nd_idx][nc_idx] = nc_data['neuroncore_utilization']

        for crt_idx in range(0, self.neuron_device_count):
            if crt_idx in nd_data:
                widget_group[crt_idx].update_data(nd_data[crt_idx])
            else:
                widget_group[crt_idx].update_data([0.0] * self.neuroncore_count)

    def _reset_neuroncore_counters(self, widget_group, *_):
        for crt_idx in range(0, self.neuron_device_count):
            widget_group[crt_idx].update_data([0.0] * self.neuroncore_count)

    def _repartition_loaded_model_data(self, model_data):
        nd_data = {}
        if model_data is None:
            return nd_data
        for model in model_data:
            for _, sg_data_obj in model['subgraphs'].items():
                nc_idx = sg_data_obj['neuroncore_index']
                nd_idx = sg_data_obj['neuron_device_index']
                if nd_idx not in nd_data:
                    nd_data[nd_idx] = {
                        'host': 0,
                        'neuron_device': 0,
                        'children': {}
                    }
                nd_obj = nd_data[nd_idx]
                if nc_idx not in nd_obj['children']:
                    nd_obj['children'][nc_idx] = {
                        'host': 0,
                        'neuron_device': 0,
                        'children': {}
                    }
                nc_obj = nd_obj['children'][nc_idx]
                sg_obj = {
                    'is_running': model['is_running'],
                    'name': model['name']
                }
                for mem_type in ['host', 'neuron_device']:
                    sg_memory_used = sg_data_obj['memory_used_bytes'][mem_type]
                    nd_obj[mem_type] += sg_memory_used
                    nc_obj[mem_type] += sg_memory_used
                    sg_obj[mem_type] = sg_memory_used
                nc_obj['children'][model['model_id']] = sg_obj
        return nd_data

    def _update_memory_used(self, widget_group, data, system_data):
        key_name = 'neuron_runtime_used_bytes'
        if system_data is not None:
            widget_group[key_name].update_data(data[key_name]['host'], system_data['memory_info']['memory_total_bytes'],
                                               data[key_name]['neuron_device'])
        key_name = 'runtime_memory_models'
        list_data = self._repartition_loaded_model_data(data['loaded_models'])
        widget_group[key_name].update_data(list_data)

    def _reset_memory_used(self, widget_group, *_):
        key_name = 'neuron_runtime_used_bytes'
        widget_group[key_name].update_data(-1, -1, 0)
        key_name = 'runtime_memory_models'
        widget_group[key_name].update_data({})

    def _update_neuron_runtime_vcpu_usage(self, widget_group, data, system_data):
        runtime_vcpu_usage = {
            'user': data['vcpu_usage']['user'],
            'system': data['vcpu_usage']['system'],
        }

        avg_system_vcpu_usage = system_data['vcpu_usage']['average_usage']
        vcpu_usage_aggregation = {
            'user': ['user', 'nice'],
            'system': ['system', 'io_wait', 'irq', 'soft_irq']
        }
        system_vcpu_usage = {}
        for name, fields in vcpu_usage_aggregation.items():
            system_vcpu_usage[name] = sum([avg_system_vcpu_usage[x] for x in fields])

        key_name = 'neuron_runtime_vcpu_usage'
        widget_group[key_name].update_data(system_vcpu_usage, runtime_vcpu_usage)

    def _reset_neuron_runtime_vcpu_usage(self, widget_group, *_):
        key_name = 'neuron_runtime_vcpu_usage'
        widget_group[key_name].update_data(None, {'user': 0, 'system': 0})

    def _set_label(self, name, pos_y, margins, texts, aligns, colors, bg_color):
        if name not in self.label_widgets:
            self.label_widgets[name] = LabelWidget(self.screen, pos_y, margins, texts, aligns, colors, bg_color)
        else:
            self.label_widgets[name].set_pos_y(pos_y)
            self.label_widgets[name].set_texts(texts, aligns)
            self.label_widgets[name].set_colors(colors, bg_color)

    def _update_header(self):
        crt_y = self.groups_next_y
        self._set_label('main_header', crt_y, (0, 0),
                        ('neuron-top', ), (LabelWidget.ALIGN_CENTER, ),
                        (curses.COLOR_WHITE, ), curses.COLOR_BLUE)
        self.groups_next_y += 2

    def _get_current_pid_index(self):
        for idx, pid in enumerate(self.available_pids):
            if pid == self.current_runtime_pid:
                return idx
        return -1

    def _update_tab_display(self):
        if not self.refresh_runtime_tab and not self.screen_resized:
            return
        current_idx = self._get_current_pid_index()
        pids_strings = [' ' + ('[%d]:' % (idx + 1) if idx < 9 else '') +
                        self.available_runtimes[pid]['neuron_runtime_tag'] + ' '
                        for idx, pid in enumerate(self.available_pids)]
        pids_colors = [curses.COLOR_WHITE for _ in range(self.RUNTIME_TAB_COUNT)]
        start, end = 0, 0
        if len(pids_strings) < self.RUNTIME_TAB_COUNT:
            pids_strings += [''] * (self.RUNTIME_TAB_COUNT - len(pids_strings))
        if current_idx != -1:
            start, end = (self.runtime_tab_offset, self.runtime_tab_offset + self.RUNTIME_TAB_COUNT - 1)
            if current_idx < start:
                start = current_idx
                end = current_idx + self.RUNTIME_TAB_COUNT - 1
            elif current_idx > end:
                end = current_idx
                start = current_idx - self.RUNTIME_TAB_COUNT + 1
            pids_colors[current_idx - start] = curses.COLOR_GREEN
            pids_strings[current_idx] = ">" + pids_strings[current_idx][1:-1] + "<"
        self.runtime_tab_offset = start
        crt_y = self.screen.get_height() - 2
        self._set_label('runtimes', crt_y, (0, 0), tuple(['Neuron Apps'] + pids_strings[start:end + 1]),
                        (LabelWidget.ALIGN_LEFT, ), tuple([curses.COLOR_YELLOW] + pids_colors),
                        curses.COLOR_BLUE)

    def _reset(self):
        self._call_widget_metric_group_func('_reset_', False)

    def _select_new_pid(self, pid):
        if self.current_runtime_pid != pid:
            self.current_runtime_pid = pid
            self.refresh_runtime_tab = True
            self.reset_ui = True

    def _select_tab(self, tab_index):
        if tab_index < len(self.available_pids):
            self._select_new_pid(self.available_pids[tab_index])

    def _select_tab_next_prev(self, go_next):
        if not self.available_pids:
            return
        current_idx = self._get_current_pid_index()
        if current_idx == -1:
            current_idx = 0
        else:
            if go_next:
                current_idx = (current_idx + 1) % len(self.available_pids)
            else:
                current_idx = (len(self.available_pids) + (current_idx - 1)) % len(self.available_pids)
        self._select_new_pid(self.available_pids[current_idx])

    def _show_usage_label(self):
        crt_y = self.screen.get_height() - 1
        self._set_label('info', crt_y, (0, 0),
                        ('q: quit',
                         'arrows: move tree selection',
                         'enter: expand/collapse tree item',
                         'x: expand/collapse entire tree',
                         'a/d: previous/next tab',
                         '1-9: select tab'), (LabelWidget.ALIGN_CENTER, ),
                        (curses.COLOR_WHITE, ), curses.COLOR_BLUE)

    def _update_footer(self):
        self._update_tab_display()
        self._show_usage_label()

    def _update_state_loading(self):
        widget_name = 'loading'
        y_pos = self.screen.get_height() // 2
        self._set_label(widget_name, y_pos, (20, 20),
                        ('LOADING', ), (LabelWidget.ALIGN_CENTER, ),
                        (curses.COLOR_WHITE, ), curses.COLOR_BLUE)
        if self.monitor_ready:
            self._change_state(NeuronMonitorTop.STATE_MAIN)

    def _update_state_main(self):
        self.groups_next_y = 1
        self._update_header()
        self._update_footer()
        if self.rebuild_ui:
            self._build_ui()
            self.rebuild_ui = False
        if self.reset_ui:
            self._reset()
            self.reset_ui = False
        if self.current_runtime_pid not in self.available_runtimes:
            return
        runtime_data = self.available_runtimes[self.current_runtime_pid]
        if runtime_data['error'] != '':
            return
        report = runtime_data['report']
        self._call_widget_metric_group_func('_update_', False, report, self.system_data)
        self.refresh_runtime_tab = False

    def _update_current_state(self):
        getattr(self, '_update_%s' % (self.state, ), lambda: None)()

    def _update(self):
        if not self.running:
            return
        self._update_current_state()

    def _handle_input(self, key):
        if not self._on_key(key):
            for target in self.key_listeners:
                target.on_key(key)

    def _on_key(self, key):
        if key == ord('q') or key == ord('Q'):
            self._stop()
            return True
        if key > ord('0') and key <= ord('9'):
            self._select_tab(key - ord('1'))
        elif key == ord('a') or key == ord('A'):
            self._select_tab_next_prev(False)
        elif key == ord('d') or key == ord('D'):
            self._select_tab_next_prev(True)
        return False

    def _register_key_listener(self, target):
        self.key_listeners.add(target)

    def _remove_all_key_listeners(self):
        self.key_listeners = set()

    def _refresh_screen(self):

        def update_widgets_in_dict(widget_dict):
            for _, widget in widget_dict.items():
                if self.screen_resized:
                    widget.on_screen_resize()
                if widget.refresh_requested():
                    widget.refresh()
                    widget.on_post_refresh()

        if self.screen_resized:
            self.screen.clear_screen()
            if self.state is NeuronMonitorTop.STATE_MAIN:
                self._update_footer()

        for _, widget_group in self.widgets.items():
            update_widgets_in_dict(widget_group)
        update_widgets_in_dict(self.label_widgets)
        self.screen.refresh()

    def _internal_loop(self):
        while self.running:
            update_type, update_arg = self.update_queue.get()
            if update_type is NeuronMonitorTop.UPDATE_TYPE_DATA:
                self._update_data(update_arg)
            elif update_type is NeuronMonitorTop.UPDATE_TYPE_KEYS:
                self._handle_input(update_arg)
            elif update_type is NeuronMonitorTop.UPDATE_TYPE_SCR_RESIZE:
                self.screen_resized = True
            if not self.running:
                break
            self._update()
            with self.screen_update_lock:
                self._refresh_screen()
            self.screen_resized = False
            if update_type is NeuronMonitorTop.UPDATE_TYPE_KEYS:
                self._on_processed_key_event()

        self.data_update_thread.join()
        self.ui_update_thread.join()

    def _stop(self):
        self.running = False
        self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_QUIT, None))


def main():
    app = NeuronMonitorTop()
    app.run()


if __name__ == '__main__':
    main()
