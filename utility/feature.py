import numpy as np


class TetrisDTFeature(object):
    def __init__(self):
        self.size = 9
        self.features = np.zeros(self.size)
        self.height = 0
        self.width = 0
        
#     def after_state(self, env, observation, action):
#         env.reset(observation[0], observation[1])
#         observation_, _reward, _done, info = self.env.step(action)
#         return observation_[0][4:,:], info
    
    def set_heights(self, after_state):
        self.heights = np.zeros(self.width)
        for i in range(self.width):
            self.heights[i] = self.height
            for j in range(self.height):
                if after_state[j,i] == 1:
                    self.heights[i] = j
                    break
            
            
            
    def row_transitions(self, after_state):
        col_0 = np.ones((self.height, 1))
        col_1 = np.ones((self.height, 1))
        board = np.hstack((col_0,after_state,col_1))
        row_transitions = np.sum(np.abs(board[:, 0:self.width+1]-board[:, 1:self.width+2]))
        return row_transitions
    
    def column_transitions(self, after_state):
        row_0 = np.zeros((1, self.width))
        row_1 = np.ones((1, self.width))
        board = np.vstack((row_0, after_state, row_1))
        column_transitions = np.sum(np.abs(board[0:self.height+1,:]-board[1:self.height+2,:]))
        return column_transitions
    
#     def max_height(self):
#         return self.height-np.min(self.heights)
    
    def holes(self, after_state):
#         index = np.arange(self.width)
#         holes = np.sum(self.height-self.heights)-np.sum(after_state[self.heights[index]:,:])
        holes = 0
        for i in range(self.width):
            if self.heights[i]<self.height:
                holes += self.height-self.heights[i]-np.sum(after_state[int(self.heights[i]):,i])
        return holes
    
    def hole_depth(self, after_state):
        hole_depth = 0
        for i in range(self.width):
            full_cells = 0
            if self.heights[i] < self.height:
                for block in after_state[int(self.heights[i]):,i]:
                    if block == 0:
                        hole_depth += full_cells
                        full_cells = 0
                    else:
                        full_cells += 1
        return hole_depth
    
    def rows_wth_holes(self, after_state):
        board = after_state.copy()
#         index = np.arange(self.width)
#         board[self.heights[index]:,:] = np.abs(board[self.heights[index]:,:]-1)
        for i in range(self.width):
            if self.heights[i]<self.height:
                board[int(self.heights[i]):,i] = np.abs(board[int(self.heights[i]):,i]-1)
        return np.sum(np.sum(board, axis=1)>0)
    
    def board_wells(self, after_state):
        board_wells = 0
        mask = np.array([1,0,1])
        col_0 = np.ones((self.height, 1))
        col_1 = np.ones((self.height, 1))
        board = np.hstack((col_0, after_state, col_1))
        for i in range(1, self.width+1):
            j = 0
            while j < self.height:
                if (board[j,i-1:i+2] == mask).all():
                    wells = 0
                    p = j
                    for k in range(j,self.height):
                        p = k
                        if board[k,i] == 1:
                            break
                        wells += 1
                    board_wells += wells*(wells+1)/2.0
                    j = p
                j += 1
        return board_wells
    
    def diversity(self, after_state):
        pattern = self.heights[0:self.width-1]-self.heights[1:self.width]
        pattern = np.array(list(set(pattern)))
        diversity = np.sum((pattern<=2)&(pattern>=-2))
        return int(diversity)
    
    def feature(self, after_state, info):
        self.height, self.width = after_state.shape
        self.set_heights(after_state)
#         print(self.heights)
        self.features[0] = info[0]*1.0
        self.features[1] = info[1]*1.0
        self.features[2] = self.row_transitions(after_state)*1.0
        self.features[3] = self.column_transitions(after_state)*1.0
        self.features[4] = self.holes(after_state)*1.0
        self.features[5] = self.hole_depth(after_state)*1.0
        self.features[6] = self.rows_wth_holes(after_state)*1.0
        self.features[7] = self.board_wells(after_state)*1.0
        self.features[8] = self.diversity(after_state)*1.0
        return self.features.copy()
    
    def feature_norm(self, after_state, info):
        features = self.feature(after_state, info).copy()
        features[0] /= 1.0*10
        features[1] /= 1.0*16
        features[2] /= 1.0*100
        features[3] /= 1.0*100
        features[4] /= 1.0*100
        features[5] /= 1.0*100
        features[6] /= 1.0*10
        features[7] /= 1.0*275
        features[8] /= 1.0*5
        return features
        
        
        