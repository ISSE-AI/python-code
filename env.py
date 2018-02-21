import numpy as np
import pyglet



# 机械臂action类
class ArmEnv(object):
    
    viewer = None
    
    dt = .1                # refresh rate 每一步关节转动用时0.1s  
    action_bound = [-1, 1] # 每一步关节可转动的角度为-1至+1度之间
    
    goal = {'x': 100., 'y': 100., 'l': 40}
    
    state_dim = 9   # 可以观测到9个状态，分别是 arm1 和 arm2 端点到 中心 和 goal 的 x,y 坐标, 最后一个信息是 finger 是否在 goal 的区域内.
                    # 之前只用到了两个手臂的转动角度信息. 按学习情况来看, 这个信息不能提供全面的state，造成收敛性差，即使得迭代次数(学习次数)增多,收敛速度变慢
    action_dim = 2  # 两个关节可移动，故设置两个action


    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])  # 产生一个2x2的表
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0

    def step(self, action):   # 传入随机的action
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize 当转角大于360度时，归一化回到0-360度

        # 计算末段位置
        (a1l, a2l) = self.arm_info['l']    # radius, arm length
        (a1r, a2r) = self.arm_info['r']    # radian, angle
        a1xy = np.array([200., 200.])      # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        # normalize features 奖励的连续性
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]    # 肘关节到目标的距离
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]  # 末段到目标的距离
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)  # reward

        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1      # 定义一个变量来保证连续停留的效果
                if self.on_goal > 50:  # 当在蓝色目标上面停留连续超过50步时才算完成任务结束当前回合
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):  # 初始化每个回合机械臂和目标的位置，下面将机械臂位置设为随机，目标位置设为固定
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2) # 将arm的转角信息设为随机数
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self): # 生成一些随机的转角，让每一个step转一下是多少度的角
        return np.random.rand(2)-0.5    # two radians





# 机械臂可视化类
class Viewer(pyglet.window.Window):     # pyglet库
    
    bar_thc = 5  # 臂宽

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)  # 白色
        
        self.arm_info = arm_info
        self.center_coord = np.array([200, 200]) # 机械臂基座放置在画布中心

        self.batch = pyglet.graphics.Batch()     # display whole batch at once
        
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))      # cc.c3b(r,g,b) = cc.4f(r/255,g/255,b/255,透明度)

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):                  # 将action通过几何关系、三角函数关系，得到绘制数据
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))





if __name__ == '__main__':
    env = ArmEnv()
    while True:                 # 可视化的主循环
        env.render()
        env.step(env.sample_action())
