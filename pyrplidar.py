import sys, time, codecs, serial, struct, multiprocessing, math

SYNC_BYTE = b'\xA5'
SYNC_BYTE2 = b'\x5A'

GET_INFO_BYTE = b'\x50'
GET_HEALTH_BYTE = b'\x52'

STOP_BYTE = b'\x25'
RESET_BYTE = b'\x40'

SCAN_BYTE = b'\x20'
FORCE_SCAN_BYTE = b'\x21'

DESCRIPTOR_LEN = 7
INFO_LEN = 20
HEALTH_LEN = 3

INFO_TYPE = 4
HEALTH_TYPE = 6
SCAN_TYPE = 129

MAX_MOTOR_PWM = 1023
DEFAULT_MOTOR_PWM = 660
SLOW_MOTOR_PWM = 60
SET_PWM_BYTE = b'\xF0'

_HEALTH_STATUSES = {
    0: 'Good',
    1: 'Warning',
    2: 'Error',
}

class RPLidarException(Exception):
    
    def __init__(self, rplidar, message):
        rplidar.stop()
        rplidar.stop_motor()
        rplidar.clear_input()    
        rplidar.reset()
        rplidar.disconnect()

class RPLidar(object):

    def __init__(self, port='COM3', baudrate=115200, timeout=1):
        self._serial_port = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.motor_running = None
        self.connect()

    def connect(self):
        if self._serial_port is not None:
            self.disconnect()
        try:
            self._serial_port = serial.Serial(
                self.port, self.baudrate,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout)
        except serial.SerialException as err:
            raise RPLidarException(self, 'Failed to connect to the sensor due to: {}'.format(err))

    def disconnect(self):
        if self._serial_port is None:
            return
        self._serial_port.close()

    def set_pwm(self, pwm):
        assert(0 <= pwm <= MAX_MOTOR_PWM)
        payload = struct.pack("<H", pwm)
        self._send_payload_cmd(SET_PWM_BYTE, payload)

    def start_motor(self, motor_speed):
        self._serial_port.setDTR(False)
        self.set_pwm(motor_speed)
        self.motor_running = True

    def stop_motor(self):
        self.set_pwm(0)
        time.sleep(.001)
        self._serial_port.setDTR(True)
        self.motor_running = False

    def _send_payload_cmd(self, cmd, payload):
        size = struct.pack('B', len(payload))
        req = SYNC_BYTE + cmd + size + payload
        checksum = 0
        for v in struct.unpack('B'*len(req), req):
            checksum ^= v
        req += struct.pack('B', checksum)
        self._serial_port.write(req)

    def _send_cmd(self, cmd):
        req = SYNC_BYTE + cmd
        self._serial_port.write(req)

    def _read_descriptor(self):
        descriptor = self._serial_port.read(DESCRIPTOR_LEN)
        if len(descriptor) != DESCRIPTOR_LEN:
            raise RPLidarException(self, 'Descriptor length mismatch')
        elif not descriptor.startswith(SYNC_BYTE + SYNC_BYTE2):
            raise RPLidarException(self, 'Incorrect descriptor starting bytes')
        is_single = descriptor[-2] == 0
        return descriptor[2], is_single, descriptor[-1]

    def _read_response(self, dsize):
        data = self._serial_port.read(dsize)
        if len(data) != dsize:
            raise RPLidarException(self, 'Wrong body size')
        return data

    def get_info(self):
        self._send_cmd(GET_INFO_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != INFO_LEN:
            raise RPLidarException(self, 'Wrong get_info reply length')
        if not is_single:
            raise RPLidarException(self, 'Not a single response mode')
        if dtype != INFO_TYPE:
            raise RPLidarException(self, 'Wrong response data type')
        raw = self._read_response(dsize)
        serialnumber = codecs.encode(raw[4:], 'hex').upper()
        serialnumber = codecs.decode(serialnumber, 'ascii')
        data = {
            'model': raw[0],
            'firmware': (raw[2], raw[1]),
            'hardware': raw[3],
            'serialnumber': serialnumber,
        }
        return data

    def get_health(self):
        self._send_cmd(GET_HEALTH_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != HEALTH_LEN:
            raise RPLidarException(self, 'Wrong get_info reply length')
        if not is_single:
            raise RPLidarException(self, 'Not a single response mode')
        if dtype != HEALTH_TYPE:
            raise RPLidarException(self, 'Wrong response data type')
        raw = self._read_response(dsize)
        status = _HEALTH_STATUSES[raw[0]]
        error_code = (raw[1] << 8) + raw[2]
        return status, error_code

    def clear_input(self):
        self._serial_port.read_all()

    def stop(self):
        self._send_cmd(STOP_BYTE)
        time.sleep(.001)
        self.clear_input()

    def reset(self):
        self._send_cmd(RESET_BYTE)
        time.sleep(.002)

    def iter_measurments(self, motor_speed, max_buf_meas=500):
        self.start_motor(motor_speed)
        status, error_code = self.get_health()
        if status == _HEALTH_STATUSES[2]:
            self.reset()
            status, error_code = self.get_health()
            if status == _HEALTH_STATUSES[2]:
                raise RPLidarException(self, 'RPLidar hardware failure. Error code: {}'.format(error_code))
        cmd = SCAN_BYTE
        self._send_cmd(cmd)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != 5:
            raise RPLidarException(self, 'Wrong get_info reply length')
        if is_single:
            raise RPLidarException(self, 'Not a multiple response mode')
        if dtype != SCAN_TYPE:
            raise RPLidarException(self, 'Wrong response data type')
        while True:
            raw = self._read_response(dsize)
            if max_buf_meas:
                data_in_buf = self._serial_port.in_waiting
                if data_in_buf > max_buf_meas*dsize:
                    self._serial_port.read(data_in_buf//dsize*dsize)
            new_scan = bool(raw[0] & 0b1)
            inversed_new_scan = bool((raw[0] >> 1) & 0b1)
            quality = raw[0] >> 2
            if new_scan == inversed_new_scan:
                raise RPLidarException(self, 'New scan flags mismatch')
            check_bit = raw[1] & 0b1
            if check_bit != 1:
                raise RPLidarException(self, 'Check bit not equal to 1')
            angle = ((raw[1] >> 1) + (raw[2] << 7)) / 64.
            distance = (raw[3] + (raw[4] << 8)) / 4.
            yield new_scan, quality, angle, distance

    def iter_scan_points(self, max_buf_meas=500, min_len=5):
        result = []
        iterator = self.iter_measurments(max_buf_meas, DEFAULT_MOTOR_PWM)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(result) > min_len:
                    yield result
                result = []
            if quality > 0 and distance > 0:
                result.append((distance*math.cos(angle*math.pi/180),
                distance*math.sin(angle*math.pi/180)))

    def iter_scans(self, max_buf_meas=500, min_len=5):
        scan = []
        iterator = self.iter_measurments(max_buf_meas, DEFAULT_MOTOR_PWM)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan) > min_len:
                    yield scan
                scan = []
            if quality > 0 and distance > 0:
                scan.append((quality, angle, distance))   

    def get_distance(self, target, max_buf_meas=500, min_len=5):
        iterator = self.iter_measurments(max_buf_meas, SLOW_MOTOR_PWM)
        for new_scan, quality, angle, distance in iterator:
            if quality > 0 and distance > 0:
                if distance == angle:
                    yield distance
                elif abs(angle-target)/360 < 0.01 or abs(angle-target-360) < 0.01:                        
                    yield distance

