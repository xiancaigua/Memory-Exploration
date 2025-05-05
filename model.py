import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
            
    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim
    

class MsgMerger(nn.Module):
    def __init__(self, embedding_dim):
        super(MsgMerger, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim
        self.query_dim = self.embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim

        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.query_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.input_dim, self.value_dim))


        self.init_parameters()


    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, msg_stacked):
        query = torch.matmul(msg_stacked, self.w_query)
        key = torch.matmul(msg_stacked, self.w_key)
        value = torch.matmul(msg_stacked, self.w_value)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.norm_factor
        
        attn_weights = torch.softmax(scores, dim=-1)

        merged_msg = torch.matmul(attn_weights, value).sum(dim=1)  

        return merged_msg


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1, mode = 'base'):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])
        if mode == 'ntm':
            self.memory_merger = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, tgt, memory, collective_memory=None, key_padding_mask=None, attn_mask=None):
        if collective_memory is not None:
            if len(collective_memory.shape) ==2:
                collective_memory = collective_memory.unsqueeze(1)  # 训练时有多个机器人的embedding形状为BATCH*1*N
            elif len(collective_memory.shape) ==1:
                collective_memory = collective_memory.unsqueeze(0).unsqueeze(0) # 跑数据时只有单个机器人的单个embedding形状为N
            tgt = self.memory_merger(torch.cat((tgt, collective_memory), dim=-1))
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w

class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim, mode = 'base'):
        super(PolicyNet, self).__init__()
        self.mode = mode

        self.node_inputs_embedding = nn.Linear(node_dim, embedding_dim)
        self.graph_node_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        self.self_state_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.cooperative_state_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1, mode= mode)

        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.pointer = SingleHeadAttention(embedding_dim)

        if mode == 'ntm':
            self.memory_input_embedding = nn.Linear(node_dim * 3, embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        graph_feature = self.node_inputs_embedding(node_inputs)
        enhanced_node_feature = self.graph_node_encoder(src=graph_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)
        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                 current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.self_state_decoder(tgt = current_node_feature,
                                                                    memory = enhanced_node_feature,
                                                                    key_padding_mask = node_padding_mask)

        return current_node_feature, enhanced_current_node_feature
    
    def merge_msg(self, msg_stacked):
        return self.msg_merger(msg_stacked)

    def decode_cooperative_state(self, current_state_feature, msg_stacked, memory_vector = None):

        enhanced_cooperative_state_feature, _, = self.cooperative_state_decoder(tgt = current_state_feature,
                                                                        memory = msg_stacked, 
                                                                        collective_memory = memory_vector)

        return enhanced_cooperative_state_feature
    
    def get_current_state_feature(self, node_inputs, node_padding_mask, edge_mask, current_index,current_coord):

        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        
        current_node_feature, current_state_feature = self.decode_state(enhanced_node_feature,
                                                                        current_index,
                                                                        node_padding_mask)
        
        current_state_feature = self.current_embedding(torch.cat((current_node_feature,
                                                                current_state_feature), dim=-1))
        
        return enhanced_node_feature, current_state_feature

    def output_policy(self, enhanced_node_feature, enhanced_cooperative_state_feature,
                      current_edge, edge_padding_mask):

        embedding_dim = enhanced_node_feature.size()[2]
        current_edge_index = current_edge.repeat(1, 1, embedding_dim)
        neighboring_feature = torch.gather(enhanced_node_feature, 1, current_edge_index.to(enhanced_node_feature.device))

        logp = self.pointer(enhanced_cooperative_state_feature, neighboring_feature, edge_padding_mask)
        logp = logp.squeeze(1)

        return logp


    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, current_coord, msg_stacked, memory_vector = None):
        
        enhanced_node_feature, current_state_feature = self.get_current_state_feature(node_inputs, node_padding_mask, edge_mask, current_index, current_coord)

        if self.mode == 'ntm':
            memory_vector = self.memory_input_embedding(memory_vector)
        enhanced_cooperative_state_feature = self.decode_cooperative_state(current_state_feature,
                                                                                     msg_stacked, 
                                                                                     memory_vector = memory_vector)
        logp = self.output_policy(enhanced_node_feature, enhanced_cooperative_state_feature,
                                  current_edge, edge_padding_mask)
        return logp


class QNet(nn.Module):
    def __init__(self, local_node_dim, embedding_dim):
        super(QNet, self).__init__()

        self.initial_local_embedding = nn.Linear(local_node_dim, embedding_dim)
        self.local_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        self.local_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.q_values_layer = nn.Linear(embedding_dim * 2, 1)

    def encode_local_graph(self, local_node_inputs, local_node_padding_mask, local_edge_mask):
        local_node_feature = self.initial_local_embedding(local_node_inputs)
        enhanced_local_node_feature = self.local_encoder(src=local_node_feature,
                                                         key_padding_mask=local_node_padding_mask,
                                                         attn_mask=local_edge_mask)

        return enhanced_local_node_feature

    def decode_local_state(self, enhanced_local_node_feature, current_local_index, local_node_padding_mask):
        embedding_dim = enhanced_local_node_feature.size()[2]
        current_local_node_feature = torch.gather(enhanced_local_node_feature, 1,
                                                  current_local_index.repeat(1, 1, embedding_dim))

        enhanced_current_local_node_feature, _ = self.local_decoder(tgt = current_local_node_feature,
                                                                    memory = enhanced_local_node_feature,
                                                                    key_padding_mask = local_node_padding_mask)
        

        return current_local_node_feature, enhanced_current_local_node_feature

    def output_q(self, current_local_node_feature, enhanced_current_local_node_feature, enhanced_local_node_feature,
                 current_local_edge, local_edge_padding_mask):
        embedding_dim = enhanced_local_node_feature.size()[2]
        k_size = current_local_edge.size()[1]
        current_state_feature = current_local_node_feature
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_local_node_feature,
                                                                 current_local_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_local_node_feature, 1,
                                           current_local_edge.repeat(1, 1, embedding_dim))

        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        q_values = self.q_values_layer(action_features)
        return q_values

    # @torch.compile
    def forward(self, local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,
                current_local_edge, local_edge_padding_mask):
        enhanced_local_node_feature = self.encode_local_graph(local_node_inputs, local_node_padding_mask, local_edge_mask)
        current_local_node_feature, enhanced_current_local_node_feature = self.decode_local_state(enhanced_local_node_feature, current_local_index, local_node_padding_mask)
        q_values = self.output_q(current_local_node_feature, enhanced_current_local_node_feature,
                                 enhanced_local_node_feature, current_local_edge, local_edge_padding_mask)

        return q_values


class NeuralTuringMachine(nn.Module):
    def __init__(self, input_size, output_size, controller_size, memory_size, 
                 memory_vector_size, num_heads=1, batch_size=1, train_device='cpu',work_device='cpu'):
        super(NeuralTuringMachine, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.controller_size = controller_size
        self.memory_size = memory_size
        self.memory_vector_size = memory_vector_size
        self.num_heads = num_heads

        self.cur_episode = 0
        self.mode = 'working'
        self.train_device = train_device
        self.work_device = work_device
        self.device = train_device

        self.memory = torch.zeros(self.memory_size, self.memory_vector_size)
        # 初始化控制器
        self.controller = nn.LSTMCell(input_size, controller_size)

        # 初始化读写头
        # 读头参数
        self.read_key_layers = nn.ModuleList([nn.Linear(controller_size, memory_vector_size) for _ in range(num_heads)])
        self.read_beta_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        self.read_gate_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        self.read_shift_layers = nn.ModuleList([nn.Linear(controller_size, 3) for _ in range(num_heads)])
        self.read_gamma_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        # 写头参数
        self.write_key_layers = nn.ModuleList([nn.Linear(controller_size, memory_vector_size) for _ in range(num_heads)])
        self.write_beta_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        self.write_gate_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        self.write_shift_layers = nn.ModuleList([nn.Linear(controller_size, 3) for _ in range(num_heads)])
        self.write_gamma_layers = nn.ModuleList([nn.Linear(controller_size, 1) for _ in range(num_heads)])
        self.write_erase_layers = nn.ModuleList([nn.Linear(controller_size, memory_vector_size) for _ in range(num_heads)])
        self.write_add_layers = nn.ModuleList([nn.Linear(controller_size, memory_vector_size) for _ in range(num_heads)])

        # 初始化输出层
        self.output_layer = nn.Linear(controller_size + memory_vector_size * num_heads, output_size)

        self.ntm_state = self.init_state(batch_size)
    def set_pretrain_mode(self, batch_size):
        self.mode = 'pretraining'
        self.device = self.train_device
        self.batch_size = batch_size
        self.ntm_state = self.init_state(self.batch_size)
        
        self.memory = self.memory.to(self.device)
        self.ntm_state['controller_state'][0] = self.ntm_state['controller_state'][0].to(self.device)
        self.ntm_state['controller_state'][1] = self.ntm_state['controller_state'][1].to(self.device)
        for i in range(self.num_heads):
            self.ntm_state['read_weights'][i]=  self.ntm_state['read_weights'][i].to(self.device)
            self.ntm_state['write_weights'][i] = self.ntm_state['write_weights'][i].to(self.device)
        self.to(self.device)
    def set_train_mode(self, batch_size):
        self.mode = 'training'
        self.device = self.train_device
        self.batch_size = batch_size
        self.ntm_state = self.init_state(self.batch_size)
        
        self.memory = self.memory.to(self.device)
        self.ntm_state['controller_state'][0] = self.ntm_state['controller_state'][0].to(self.device)
        self.ntm_state['controller_state'][1] = self.ntm_state['controller_state'][1].to(self.device)
        for i in range(self.num_heads):
            self.ntm_state['read_weights'][i]=  self.ntm_state['read_weights'][i].to(self.device)
            self.ntm_state['write_weights'][i] = self.ntm_state['write_weights'][i].to(self.device)
        self.to(self.device)
    def set_work_mode(self):
        self.mode = 'working'
        self.device = self.work_device
        self.batch_size = 1
        self.ntm_state = self.init_state(self.batch_size)
        
        self.memory = self.memory.to(self.device)
        self.ntm_state['controller_state'][0] = self.ntm_state['controller_state'][0].to(self.device)
        self.ntm_state['controller_state'][1] = self.ntm_state['controller_state'][1].to(self.device)
        for i in range(self.num_heads):
            self.ntm_state['read_weights'][i]=  self.ntm_state['read_weights'][i].to(self.device)
            self.ntm_state['write_weights'][i] = self.ntm_state['write_weights'][i].to(self.device)
        self.to(self.device)
    def init_state(self, batch_size):

        controller_state = [torch.zeros(batch_size, self.controller_size).to(self.device),
                            torch.zeros(batch_size, self.controller_size).to(self.device)]
        read_weights = [torch.zeros(batch_size, self.memory_size).to(self.device) for _ in range(self.num_heads)]
        write_weights = [torch.zeros(batch_size, self.memory_size).to(self.device) for _ in range(self.num_heads)]
        return {
            'controller_state': controller_state,
            'read_weights': read_weights,
            'write_weights': write_weights
        }
    def set_state(self, state):
        for i in range(self.batch_size):
            self.ntm_state['controller_state'][0][i] = state[i]['controller_state'][0]
            self.ntm_state['controller_state'][1][i] = state[i]['controller_state'][1]
            for j in range(self.num_heads):
                self.ntm_state['read_weights'][j][i] = state[i]['read_weights'][j]
                self.ntm_state['write_weights'][j][i] = state[i]['write_weights'][j]
    def get_state(self):
        state = {}
        state['controller_state'] = [self.ntm_state['controller_state'][0].detach(),
                                     self.ntm_state['controller_state'][1].detach()]
        state['read_weights'] = [self.ntm_state['read_weights'][i].detach() for i in range(self.num_heads)]
        state['write_weights'] = [self.ntm_state['write_weights'][i].detach() for i in range(self.num_heads)]
        return state
    def forward(self, now_embedding):
        # now_embedding = now_embedding.to(self.device) # Batch x Input
        self.ntm_state['controller_state'][0],self.ntm_state['controller_state'][1] = self.controller(now_embedding, 
                                                             self.ntm_state['controller_state'])
        reads = self.read()
        if self.mode != 'training':
            self.write()
        output = self.output_layer(torch.cat([self.ntm_state['controller_state'][0], reads], dim=1))
        return output
    def read(self):
        read_vectors = []
        for i in range(self.num_heads):
            h = self.ntm_state['controller_state'][0]
            key = self.read_key_layers[i](h)
            beta = F.softplus(self.read_beta_layers[i](h))
            gate = torch.sigmoid(self.read_gate_layers[i](h))
            shift = F.softmax(self.read_shift_layers[i](h), dim=1)
            gamma = 1 + F.softplus(self.read_gamma_layers[i](h))

            weight = self._address_memory(self.ntm_state['read_weights'][i], key, beta, gate, shift, gamma)
            self.ntm_state['read_weights'][i] = weight
            read_vector = torch.bmm(weight.unsqueeze(1), 
                                    self.memory.detach().unsqueeze(0).expand(self.batch_size,-1,-1)).squeeze(1)
            read_vectors.append(read_vector)
        read_vectors = torch.cat(read_vectors, dim=1) 

        return read_vectors
    def write(self):
        batched_memories = torch.zeros(self.num_heads, self.batch_size, 
                                       self.memory_size, self.memory_vector_size).to(self.device)
        batched_memory = self.memory.detach().unsqueeze(0).expand(self.batch_size,-1,-1)
        for i in range(self.num_heads):
            h = self.ntm_state['controller_state'][0]
            key = self.write_key_layers[i](h)
            beta = F.softplus(self.write_beta_layers[i](h))
            gate = torch.sigmoid(self.write_gate_layers[i](h))
            shift = F.softmax(self.write_shift_layers[i](h), dim=1)
            gamma = 1 + F.softplus(self.write_gamma_layers[i](h))

            weight = self._address_memory(self.ntm_state['write_weights'][i], key, beta, gate, shift, gamma)
            self.ntm_state['write_weights'][i] = weight

            erase = torch.sigmoid(self.write_erase_layers[i](h))
            add = self.write_add_layers[i](h)

            batched_memories[i] = batched_memory * (1 - weight.unsqueeze(2) * erase.unsqueeze(1)) + \
                          weight.unsqueeze(2) * add.unsqueeze(1)
        # 将所有头的记忆合并
        mean_batched_memories = torch.mean(batched_memories, dim=0)
        self.memory = torch.mean(mean_batched_memories, dim=0).detach()

    def update_memory(self, new_memory):
        self.memory = new_memory
    def _address_memory(self, prev_weight, key, beta, gate, shift, gamma):
        content_weight = self._content_addressing(key, beta)
        weight = gate * content_weight + (1 - gate) * prev_weight
        weight = self._circular_convolution(weight, shift)
        weight = self._sharpen(weight, gamma)
        return weight
    def _content_addressing(self, key, beta):
        similarity = F.cosine_similarity(self.memory.unsqueeze(0).expand(self.batch_size,-1,-1), key.unsqueeze(1), dim=2)
        return F.softmax(beta * similarity, dim=1)
    def _circular_convolution(self, weight, shift):

        # 构建三个方向的移位矩阵
        shift_weights = []
        for s in [-1, 0, 1]:
            shifted = torch.roll(weight, shifts=s, dims=1)
            shift_weights.append(shifted.unsqueeze(2))  # [B, N, 1]
        
        # 拼接为 [B, N, 3]
        shift_stack = torch.cat(shift_weights, dim=2)

        # shift 是 [B, 3]，我们想把它广播乘以 shift_stack，然后 sum over dim=2
        shifted_result = torch.sum(shift_stack * shift.unsqueeze(1), dim=2)  # [B, N]
        return shifted_result
    def _sharpen(self, weight, gamma):
        weight = weight ** gamma
        return weight / (weight.sum(dim=1, keepdim=True) + 1e-6)
    def inspect_memory(self, topk=5):
        print("Memory Norm (avg):", self.memory.norm(dim=1).mean().item())
        top_slots = self.memory.norm(dim=1).topk(topk)
        print("Top activated memory slots:", top_slots)


    

if __name__ == '__main__':
    # 定义模型参数
    input_size = 16
    output_size = 16
    controller_size = 80
    memory_size = 20
    memory_vector_size = 16
    num_heads = 2
    batch_size = 1
    sequence_length = 128 * 20

    # 初始化模型
    ntm = NeuralTuringMachine(input_size, output_size, controller_size, memory_size,
                               memory_vector_size, num_heads, batch_size)
    # ntm = ntm.cuda()  # 如果有 GPU 可用
    ntm.set_work_mode()

    # 初始化输入
    inputs = torch.randn(batch_size, sequence_length, input_size)

    # 测试前向传播
    for t in range(sequence_length):
        now_embedding = inputs[:, t, :]
        print(f"Step {t + 1}, Input Shape: {now_embedding.shape}")
        output = ntm(now_embedding)
        print(f"Step {t + 1}, Output Shape: {output.shape}")
    