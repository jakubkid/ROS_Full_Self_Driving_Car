from lowpass import LowPassFilter
from yaw_controller import YawController
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicleMass, breakDeadband, decLimit, accLimit,
                       wheelRadius, wheelBase, steerRatio, maxLatAcc, maxSteerAngle):
        kp = 0.3
        ki = 0.1
        kd = 0.05
        minThr = 0. #Minimum throttle value
        maxThr = 1. #Maximum Throttle value
        self.throttleControler = PID(kp, ki, kd, minThr, maxThr)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.velLpf = LowPassFilter(tau, ts)

        self.yawController = YawController(wheelBase,steerRatio, 0.1, maxLatAcc, maxSteerAngle)

        self.vehicleMass = vehicleMass
        self.breakDeadband = breakDeadband
        self.decLimit = decLimit
        self.accLimit = accLimit
        self.wheelRadius = wheelRadius
        self.timeStamp = rospy.get_time()

    def control(self, dwbEn, curVel, setVel, setAngVel):
        if not dwbEn:
            self.throttleControler.reset()
            return 0., 0., 0.
        curVelFil = self.velLpf.filt(curVel)
        
        steering = self.yawController.get_steering(setVel, setAngVel, curVelFil)
        
        velError = setVel - curVelFil

        curTime = rospy.get_time()
        deltaTime = curTime - self.timeStamp
        self.timeStamp = curTime

        throttle = self.throttleControler.step(velError, deltaTime)
        brake = 0
        
        if setVel == 0. and curVelFil < 0.1:
            #We are stopped hold the break
            throttle = 0;
            brake = 400 #Nm - to hold the car in place
        # TODO this could be done by converting PID output to Nm?
        elif throttle < 0.1 and velError < 0:
            throttle = 0
            decel = max(velError, self.decLimit)
            brake = abs(decel) * self.vehicleMass * self.wheelRadius # breaking in N*m

        return throttle, brake, steering
