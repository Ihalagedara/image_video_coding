
import numpy as np
import numpy as numpy
import cv2
from scipy.fftpack import dct, idct
import math
import heapq
import numpy as np
from collections import defaultdict
import pickle
from pympler import asizeof




  
class ImageVideoCoding:
    def pad_array(self, array):
        shape = array.shape

        # Calculate the amount of padding needed for each dimension
        pad_width = []
        for dim in shape:
            remainder = dim % 8
            if remainder == 0:
                pad = 0
            else:
                pad = 8 - remainder
            pad_width.append((0, pad))  # (before_padding, after_padding)

        # Pad the array
        padded_array = np.pad(array, pad_width, mode='constant')

        return padded_array

    
    def blockshaped(self, arr, block_size):
        height = arr.shape[0]
        width = arr.shape[1]
        blocks = []

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                block = [arr[r][col:col + block_size] for r in range(row, row + block_size)]
                blocks.append(block)

        return np.array(blocks)
    
    def de_blocked(self,blocks,ori_shape):
        block_size = blocks.shape[0]
        num_blocks_row = blocks.shape[1]
        num_blocks_col = blocks.shape[2]

        height = int(np.sqrt(block_size) * num_blocks_row)
        width = int(np.sqrt(block_size) * num_blocks_col)

        arranged_image = np.zeros(ori_shape)
        k = 0
        for i in range(int(ori_shape[0]/num_blocks_row)):
            for j in range(int(ori_shape[1]/num_blocks_col)):
                arranged_image[i*num_blocks_row:i*num_blocks_row+num_blocks_row, j*num_blocks_col:j*num_blocks_col+num_blocks_col] = blocks[k]
                #print(blocks[k])
                k=k+1

        return arranged_image
        
    
    def blockshaped1(self,arr, nrows):
        z = int(arr.shape[0] * arr.shape[1] / (nrows * nrows))
        return arr.reshape(z, nrows, nrows)
    
    

    def dct_all(self, arr):
        dummy = np.zeros(arr.shape, dtype=float)
        for i in range(arr.shape[0]):            dummy[i] = dctn(arr[i],1)  # as
        return dummy

    def inv_dct_all(self, arr):
        dummy = np.zeros(arr.shape, dtype=float)
        for i in range(arr.shape[0]):
            dummy[i] = idctn(arr[i],1)  # as
        return dummy

    qunt = [[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]]

    def quant_all(self, arr, quality):
        dummy = np.zeros(arr.shape, dtype=float)
        for i in range(arr.shape[0]):
            dummy[i] = (arr[i] / self.qunt) / quality
        return dummy

    def inv_quant_all(self, arr, quality):
        dummy = np.zeros(arr.shape, dtype=float)
        for i in range(arr.shape[0]):
            dummy[i] = (arr[i] * self.qunt) * quality *3000
        return dummy

    def rearrange(self, arr):
        dummy = np.zeros(arr.shape, dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):

                if arr[i][j] > 0:
                    dummy[i][j] = math.floor(arr[i][j])
                else:
                    dummy[i][j] = math.ceil(arr[i][j])

        return dummy

    def save_file(self, my_dict, file_name):
        data = pickle.dumps(my_dict)

        with open(str(file_name + ".bin"), 'wb') as file:
            file.write(data)

    def make_dict(self, huffman_codes, encode, img_shape):
        encoded_data = encode
        padding_length = 8 - len(encoded_data) % 8
        encoded_data += "0" * padding_length
        padding_info = format(padding_length, '08b')
        encoded_data = padding_info + encoded_data

        byte_array = bytearray()
        for i in range(0, len(encoded_data), 8):
            byte = encoded_data[i:i + 8]
            byte_array.append(int(byte, 2))

        my_dict = {
            "shape": img_shape,
            "huff_code": huffman_codes,
            "encode": byte_array
        }

        return my_dict

    def load_file(self, file_name):
        with open(str(file_name + ".bin"), 'rb') as file:
            compressed_data = file.read()

        dictionary = pickle.loads(compressed_data)
        binary_data = ""

        for byte in dictionary["encode"]:
            binary_data += format(byte, '08b')

        padding_length = int(binary_data[:8], 2)
        binary_data = binary_data[8:-padding_length]

        return (
            dict(dictionary["huff_code"]),
            binary_data,
            dictionary["shape"],
            dictionary['quality']
        )

    def load_file_vid(self, file_name):
        huff = {}
        encode = []
        shape = []
        mv = []
        quality = []
        with open(str(file_name + ".bin"), 'rb') as file:
            compressed_data = file.read()

        dictionary = pickle.loads(compressed_data)
        frames = int(dictionary["frame0"]["frames"])
        shape = dictionary["frame0"]["shape"]
        for i in range(len(dictionary)):
            binary_data = ""
            frame_number = i
            key = f"frame{frame_number}"
            if key in dictionary and "encode" in dictionary[key]:
                encode_data = dictionary[key]["encode"]
                for byte in encode_data:
                    binary_data += format(byte, '08b')

            padding_length = int(binary_data[:8], 2)
            binary_data = binary_data[8:-padding_length]
            encode.append(binary_data)

            if key in dictionary and "huff_code" in dictionary[key]:
                huff_code = dictionary[key]["huff_code"]
                huff["frame" + str(i)] = huff_code

            if key in dictionary and "mv" in dictionary[key]:
                mv_get = dictionary[key]["mv"]
                mv.append(mv_get)

            if key in dictionary and "quality" in dictionary[key]:
                quality_get = dictionary[key]["quality"]
                quality.append(quality_get)

        return dict(huff), encode, shape, frames, mv, quality

    def img_set(self, img, quality):
        img1 = self.blockshaped(img,8)
        img2 = self.dct_all(img1)
        img3 = self.quant_all(img2, quality)
        img4 = self.de_blocked(img3,img.shape)
        img5 = self.rearrange(img4)
        return img5

    def dimg_set(self, dcom, shape, quality):
        img6 = self.blockshaped(dcom, 8)  # deded
        img7 = self.inv_quant_all(img6, quality)
        img8 = self.inv_dct_all(img7)
        img9 = self.de_blocked(img8,dcom.shape)                        #img8.reshape(shape)
        return img9

    def get_pixel_frequencies(self, image):
        frequencies = defaultdict(int)
        for pixel in image.flatten():
            frequencies[pixel] += 1
        return frequencies

    def build_huffman_tree(self, frequencies):
        priority_queue = []
        for pixel, frequency in frequencies.items():
            node = HuffmanNode(frequency, pixel)
            heapq.heappush(priority_queue, node)

        while len(priority_queue) > 1:
            left_child = heapq.heappop(priority_queue)
            right_child = heapq.heappop(priority_queue)
            new_frequency = left_child.frequency + right_child.frequency
            parent = HuffmanNode(new_frequency)
            parent.left = left_child
            parent.right = right_child
            heapq.heappush(priority_queue, parent)

        return priority_queue[0]

    def generate_huffman_codes(self, node, current_code, huffman_codes):
        if node.pixel is not None:
            huffman_codes[node.pixel] = current_code
            return

        self.generate_huffman_codes(node.left, current_code + "0", huffman_codes)
        self.generate_huffman_codes(node.right, current_code + "1", huffman_codes)

    def compress(self, image):
        frequencies = self.get_pixel_frequencies(image)
        huffman_tree = self.build_huffman_tree(frequencies)
        huffman_codes = {}
        self.generate_huffman_codes(huffman_tree, "", huffman_codes)

        encoded_data = ""
        for pixel in image.flatten():
            encoded_data += huffman_codes[pixel]

        return huffman_codes, encoded_data

    def decompress(self, byte_array, original_shape, huffman_codes):
        huffman_codes = {value: key for key, value in huffman_codes.items()}
        decoded_data = []
        current_code = ""
        for bit in byte_array:
            current_code += bit
            if current_code in huffman_codes:
                decoded_data.append(huffman_codes[current_code])
                current_code = ""

        decompressed_image = np.array(decoded_data)
        decompressed_image = np.pad(
            decompressed_image,
            (0, original_shape[0] * original_shape[1] - decompressed_image.shape[0]),
            mode='constant'
        )
        decompressed_image =   decompressed_image.reshape(original_shape)

        return decompressed_image

    def cal_size(self, img, n):
        huffman_codes, encode = self.compress(self.img_set(img, n))
        dic = self.make_dict(huffman_codes, encode, img.shape)
        size = asizeof.asizeof(dic) / 1000
        return size

    def set_bitrate(self, bit, img, upper, lower, limit):
        size = self.cal_size(img, lower)
        while (lower - upper) > limit:
            mid = (upper + lower) / 2
            size_mid = self.cal_size(img, mid)
            size_up = self.cal_size(img, upper)
            size_low = self.cal_size(img, lower)
            if size_mid < bit and size_low < bit:
                lower = mid
                size = self.cal_size(img, lower)
            elif bit < size_up and bit < size_mid:
                upper = mid
                size = self.cal_size(img, upper)
            else:
                break
        return (lower + upper) / 2

    def get_mc(self, current, previous):
        curr_1 = self.blockshaped(current, 8)
        pre_1 = self.blockshaped(previous, 8)
        mc = []

        for i in range(curr_1.shape[0]):
            curr_one = np.tile(curr_1[i], (curr_1.shape[0], 1, 1))
            diff = curr_one - pre_1

            diff_new = np.absolute(diff)
            mv = list(np.sum(np.sum(diff_new, axis=1), axis=1))

            if mv[i] == min(mv):
                mc.append(i)
            else:
                mc.append(mv.index(min(mv)))
        return mc

    def send_mc(self, mc1):
        send = []
        for i in range(len(mc1)):
            if mc1[i] != i:
                send.append((i, mc1[i]))
        return send

    def img_compress(self, input_name: str, output_name: str, qunt_level=1.0):
        gray_image = cv2.imread(str(input_name) + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = np.asarray(gray_image)
        img = self.pad_array(img)
        huffman_codes, encode = self.compress(self.img_set(img, qunt_level))
        dic = self.make_dict(huffman_codes, encode, img.shape)
        dic['quality'] = qunt_level
        self.save_file(dic, str(output_name))

    def img_compress_bitrate(self, input_name: str, output_name: str, bit_rate):
        gray_image = cv2.imread(str(input_name) + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = np.asarray(gray_image)
        img = self.pad_array(img)
        qunt_level = self.set_bitrate(bit_rate, img, 0.0001, 200, 0.0001)
        huffman_codes, encode = self.compress(self.img_set(img, qunt_level))
        dic = self.make_dict(huffman_codes, encode, img.shape)
        dic['quality'] = qunt_level
        self.save_file(dic, str(output_name))

    def img_decompress(self, input_name: str, output_name: str):
        code, data, shape, quality = self.load_file(str(input_name))
        decom = self.decompress(data, shape, code)
        output_img = self.dimg_set(decom, shape, quality)
        result = cv2.normalize(output_img, dst=None, alpha=0, beta=255,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(str(output_name) + '.jpg', result)
      
    def vid_compress(self,Vid_name:str,output_name:str,frames_no=0,quality=1):
        video = cv2.VideoCapture(Vid_name)

        if not video.isOpened():
            print("Error opening video file")
            exit()

        ret, frame = video.read()
        frames = []
        count = 1
        while ret:
            count = count+1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_array = np.array(frame)
            frames.append(frame_array)
            ret, frame = video.read()
        video.release()
        frames_array = np.array(frames)
        if frames == 0:
            frame_test = frames_array[0:int(count)]
        else:
            frame_test = frames_array[0:int(frames_no)]
        
        first = self.pad_array(frame_test[0])
        send_dict = {}
        quality = quality
        huffman_codes, encode = self.compress(self.img_set(first,quality))
        dic = self.make_dict(huffman_codes, encode, first.shape)
        dic["frames"] = frame_test.shape[0]
        dic['quality'] = quality
        send_dict = {"frame0" : dic}
        
        for j in range(frame_test.shape[0]-1):
            dic = {}
            current1 = self.pad_array(frame_test[j+1])
            previous1 = self.pad_array(frame_test[0])
            mc = self.get_mc(current1,previous1)
            send =  self.send_mc(mc)
            pre_block = self.blockshaped(previous1,8)
            recon_img = np.zeros(pre_block.shape)

            for i in range(recon_img.shape[0]):
                recon_img[i] = pre_block[mc[i]]

            recon_img1 = self.de_blocked(recon_img,previous1.shape)               #recon_img.reshape(current1.shape)
            residual = current1 - recon_img1
            quality = quality
            huffman_codes, encode = self.compress(self.img_set(residual,quality))
            dic = self.make_dict(huffman_codes, encode, first.shape)
            dic["mv"] = send
            dic['quality'] = quality

            send_dict["frame"+str(j+1)] = dic
            
            self.save_file(send_dict, output_name)
            
    def vid_decompress(self,file_name):
        code, data, shape, frames, mv, quality = self.load_file_vid(file_name)
        decom_vid = np.zeros((frames , shape[0],shape[1]))  #residual
        final = np.zeros((frames , shape[0],shape[1]))      #final
        mv_img = np.zeros((frames-1 , shape[0],shape[1]))   #reconstructed

        for i in range(frames):
            decom = self.decompress(data[i], shape, code["frame"+str(i)])
            decom_vid[i] = self.dimg_set(decom,shape,quality[i])
        
        img = self.blockshaped(decom_vid[0],8)

        real_mv = []
        new_mv = mv
        for j in range(len(mv)):
            tem_mv = []
        for k in range(img.shape[0]):
            tem_mv.append((k,k))
        da = np.array(tem_mv)
        for i in range(len(new_mv[j])):
            da[new_mv[j][i][0]][1] = new_mv[j][i][1]
        real_mv.append((da))
        
        recon_img1 = np.zeros(img.shape)
        for j in range(len(real_mv)):
            
            for i in range(recon_img1.shape[0]):
                recon_img1[i] = img[real_mv[j][i][1]]
            mv_img[j] = self.de_blocked(recon_img1,decom_vid[0].shape)                         #recon_img1.reshape(decom_vid[0].shape)
        
        final[0] = decom_vid[0]
        for i in range(1,len(real_mv)):
            final[i] = mv_img[i] + decom_vid[i]
        
        return final



class HuffmanNode:
  def __init__(self, frequency, pixel=None):
      self.frequency = frequency
      self.pixel = pixel
      self.left = None
      self.right = None

  def __lt__(self, other):
      return self.frequency < other.frequency
    
    