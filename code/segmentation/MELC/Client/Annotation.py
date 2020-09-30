# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

import numpy as np
import re

from xml.dom import minidom
import xml.etree.ElementTree as ET
from copy import deepcopy

class SVGAnnot:
    def __init__(self, path_svg):
        # These few lines fixes the bug during writting of svgs. Erases all string sections xlink: because it messes up the parsing below
        fid = open(path_svg, 'r')
        text = fid.read()
        fid.close()
        pos = text.find('xlink:')
        text = text[:pos] + text[pos + 'xlink:'.__len__():]

        fid = open(path_svg, 'w')
        fid.write(text)
        fid.close()

        print(path_svg)

        self.annotations = list()
        self.xmldoc = minidom.parse(path_svg)
        self.get_polygons()
        self.get_paths()


    def get_contours(self):
        contours = deepcopy(self.annotations)
        for k in range(contours.__len__()):
            annot = contours[k]
            cont = np.zeros((annot.shape[0], 1, 2), dtype=np.int32) #!!!!! otherwise do not work printing of contours in opencv MUST BE int32 (maybe uint32)
            cont[:, 0, 0] = annot[:, 0]
            cont[:, 0, 1] = annot[:, 1]
            contours[k] = cont
        return contours


    def get_polygons(self):
        itemlist = self.xmldoc.getElementsByTagName('polygon')
        for item in itemlist:
            temp_array = np.array(item.attributes['points'].value.split(' '))
            temp_array = temp_array[temp_array != '']

            annot = np.zeros((temp_array.shape[0], 2), dtype=np.uint16)
            for i in range(annot.shape[0]):
                temp = temp_array[i]
                temp = np.array(temp.split(',')).astype(np.float32).round()
                temp[temp < 0] = 0
                temp = temp.astype(np.int32)
                annot[i, 0] = temp[0]
                annot[i, 1] = temp[1]
            self.annotations.append(annot)

    def get_paths(self):
        itemlist = self.xmldoc.getElementsByTagName('path')
        for item in itemlist:
            svgpath = item.attributes['d'].value
            SVGPath_tool = SVGPath(svgpath)
            self.annotations.append(SVGPath_tool.get_polygon().round().astype(np.uint16))

class SVGPath:
    def __init__(self, svgpath):
        self.commands = list()
        self.current_position = np.zeros(2, dtype=np.float32)
        self.polygon = np.zeros((1, 2), dtype=np.float32)

        command_list = 'mlhvcsqtaz'
        split = command_list + command_list.upper()
        split_reg = command_list[0]
        path_commands = ''
        for k in range(1, len(split)):
            split_reg = split_reg + '|' + split[k]

        svgpath_attributes = np.array(re.split(split_reg, svgpath))
        svgpath_attributes = svgpath_attributes[svgpath_attributes != '']
        svgpath_commands = ''
        for letter in svgpath:
            if letter.lower().islower():
                svgpath_commands = svgpath_commands + letter

        for k in range(len(svgpath_commands)-1):
            self.set_command(svgpath_commands[k], svgpath_attributes[k])

    def set_command(self, command, str_attrs):
        if not command.lower() == 'z':
            str_attrs = np.array(str_attrs.replace('-', ',-').split(','))
            str_attrs = str_attrs[str_attrs != ''].astype(np.float32)
            com_dict = {
                'command': command,
                'attributes': str_attrs
            }
            self.commands.append(com_dict)

    def get_polygon(self):
        # decoding implemented according
        # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d
        for command in self.commands:
            if command['command'].lower() == 'm':
                self.run_m(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'l':
                self.run_l(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'h':
                self.run_h(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'v':
                self.run_v(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'c':
                self.run_c(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 's':
                self.run_s(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'q':
                self.run_q(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 't':
                self.run_t(command['command'].islower(), command['attributes'])
            elif command['command'].lower() == 'a':
                self.run_a(command['command'].islower(), command['attributes'])

        if (self.polygon[-1, 0] == self.polygon[0, 0]) and (self.polygon[-1, 1] == self.polygon[-1, 1]):
            self.polygon = self.polygon[:-1, :]

        cntr = 0
        for k in range(self.polygon.shape[0] - 1):
            if (self.polygon[k-cntr, 0] == self.polygon[k+1-cntr, 0]) and \
                    (self.polygon[k-cntr, 1] == self.polygon[k+1-cntr, 1]):
                self.polygon = self.polygon[np.arange(self.polygon.shape[0])!=k-cntr, :]
                cntr += 1

        return self.polygon


    def update_polygon(self):
        self.current_position[self.current_position < 0] = 0
        self.polygon = np.concatenate((self.polygon, self.current_position.reshape((1, 2))))
        self.current_position = self.current_position.reshape(2)

    def run_m(self, lower, x):
        if lower:
            dxdy = x
            self.current_position[0] += dxdy[0]
            self.current_position[1] += dxdy[1]
        else:
            xy = x
            self.current_position[0] = xy[0]
            self.current_position[1] = xy[1]
        self.polygon = deepcopy(self.current_position).reshape((1, 2))


    def run_l(self, lower, x):
        if lower:
            dxdy = x
            xy = self.current_position + dxdy
            #xy[0] = xy[0] + dxdy[0]
            #xy[1] = xy[0] +dxdy[1]
        else:
            xy = x

        self.current_position = xy
        self.update_polygon()

    def run_h(self, lower, x):
        if lower:
            dx = x
            x = self.current_position[0] + dx[0]
        else:
            x = x

        self.current_position[0] = x
        self.update_polygon()

    def run_v(self, lower, x):
        if lower:
            dy = x
            y = self.current_position[1] + dy[0]
        else:
            y = x
        self.current_position[1] = y
        self.update_polygon()

    def run_c(self, lower, x):
        if lower:
            dx1dy1 = x[0:2]
            dx2dy2 = x[2:4]
            dxdy = x[4:]

            x1y1 = deepcopy(self.current_position) + dx1dy1.reshape((1, 2))
            x2y2 = deepcopy(self.current_position) + dx2dy2.reshape((1, 2))
            xy = deepcopy(self.current_position) + dxdy.reshape((1, 2))

        else:
            x1y1 = x[0:2]
            x2y2 = x[2:4]
            xy = x[4:]

        self.current_position = x1y1
        self.update_polygon()
        self.current_position = x2y2
        self.update_polygon()
        self.current_position = xy
        self.update_polygon()

    def run_s(self, lower, x):
        if lower:
            dx2dy2 = x[0:2]
            dxdy = x[2:]

            x2y2 = deepcopy(self.current_position) + dx2dy2
            xy = deepcopy(self.current_position) + dxdy

        else:
            x2y2 = x[0:2]
            xy = x[2:]

        self.current_position = x2y2
        self.update_polygon()
        self.current_position = xy
        self.update_polygon()

    def run_q(self, lower, x):
        if lower:
            dx1dy1 = x[0:2]
            dxdy = x[2:4]

            x1y1 = deepcopy(self.current_position) + dx1dy1
            xy = deepcopy(self.current_position) + dxdy

        else:
            x1y1 = x[0:2]
            xy = x[2:4]

        self.current_position = x1y1
        self.update_polygon()
        self.current_position = xy
        self.update_polygon()

    def run_t(self, lower, x):
        if lower:
            dxdy = x
            self.current_position[0] += dxdy[0]
            self.current_position[1] += dxdy[1]
        else:
            xy = x
            self.current_position[0] = xy[0]
            self.current_position[1] = xy[1]
        self.update_polygon()

    def run_a(self, lower, x):
        raise Exception('Eliptical curve implementation is not implemented')














