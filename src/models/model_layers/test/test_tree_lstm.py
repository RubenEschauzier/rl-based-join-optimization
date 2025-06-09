import pytest
import torch

from src.models.model_layers.tree_lstm import ChildSumTreeLSTM, NAryTreeLSTM


class TestChildSumTreeLSTM:

    @pytest.fixture
    def tree_lstm(self):
        # Called before each test method
        in_dim = 2
        out_dim = 2
        layer = ChildSumTreeLSTM(in_dim, out_dim)

        # Define fixed weights as a dict
        weight_dict = {
            'W_iou': [torch.ones(out_dim * 3, in_dim), torch.zeros(out_dim*3)],
            'U_iou': [torch.ones(out_dim * 3, in_dim), torch.zeros(out_dim*3)],
            'W_f': [torch.ones(out_dim, in_dim), torch.zeros(out_dim)],
            'U_f': [torch.ones(out_dim, out_dim), torch.zeros(out_dim)],
        }
        layer.set_weights(weight_dict)
        return layer

    def test_message_child_sum(self, tree_lstm):
        x = torch.tensor([
            [1.0, 0],
            [.5, .5],
            [2, 2]
        ])
        #   2
        #  / \
        # 0   1  (0 -> 2) (1 -> 2)
        edges = torch.tensor([[0, 1],[2, 2]])
        x_j = torch.tensor([[1, 0], [.5, .5]])
        x_i = torch.tensor([[2, 2], [2, 2]], dtype=torch.float)
        # Suppose a hidden state IS defined
        h_j = torch.tensor([[1, 0], [0, 0]], dtype=torch.float)
        c_j = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        index = torch.tensor([2, 2])
        expected_f_c = torch.tensor([[.9933, 0], [0, 0.9820]])

        h_j_updated, f_c = tree_lstm.message(x_j, x_i, h_j, c_j, index)

        torch.testing.assert_close(h_j_updated, h_j, atol=1e-4, rtol=0)
        torch.testing.assert_close(f_c, expected_f_c, atol=1e-4, rtol=0)

    def test_get_iou_child_sum(self, tree_lstm):
        x = torch.tensor([
            [0, 1],
            [1, 0],
            [.5, .5]
        ])
        h_tilde = torch.tensor([
            [.25, .25],
            [.5, .5],
            [.75, .75]
        ], dtype=torch.float)
        expected_non_sigmoid_i_o_u = torch.tensor([
            [1.5, 1.5],
            [2, 2],
            [2.5, 2.5]
        ])
        i, o, u = tree_lstm.get_iou(x, h_tilde)
        torch.testing.assert_close(i, torch.sigmoid(expected_non_sigmoid_i_o_u), atol=1e-4, rtol=0)
        torch.testing.assert_close(o, torch.sigmoid(expected_non_sigmoid_i_o_u), atol=1e-4, rtol=0)
        torch.testing.assert_close(u, torch.tanh(expected_non_sigmoid_i_o_u), atol=1e-4, rtol=0)

    def test_get_h_c_child_sum(self, tree_lstm):
        i = torch.tensor([[0,1], [1, 0], [.5, .5]])
        o = torch.tensor([[0,1], [1, 0], [.5, .5]])
        u = torch.tensor([[0,1], [1, 0], [.5, .5]])

        f_c_sum = torch.tensor([[.25, .25], [.5, .5], [.75, .75]])
        h, c = tree_lstm.get_h_c(i, o, u, f_c_sum)
        expected_h = torch.tensor([[0, 0.8483], [0.9051, 0], [0.3808, 0.3808]])
        torch.testing.assert_close(h, expected_h, atol=1e-4, rtol=0)
        pass

    def test_forward_pass_child_sum(self, tree_lstm):
        x = torch.tensor([
            [1.0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [.5, 0],
            [2, 2]
        ])
        h = torch.zeros(x.shape)
        c = torch.zeros(x.shape)
        edges = torch.tensor([[0, 1, 2, 3, 4, 5],[4, 4, 5, 5, 6, 6]])
        #      6
        #    /   \
        #   4     5
        #  / \   / \
        # 0   1 2   3
        # Hierarchical application of tree-lstm, so first the edges to 4 and 5 then to 6
        order_parent_receive = [torch.tensor([0, 0, 0, 0, 1, 1, 0], dtype=torch.bool),
                                torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)]
        expected_c = torch.tensor([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [.5568, .5568],
            [.2876, .2876],
            [1.8296, 1.8296]
        ])
        expected_h = torch.tensor([
            [0,0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0.3696, 0.3696],
            [0.1743, 0.1743],
            [0.9440, 0.9440],
        ])
        h, c = tree_lstm.forward(x, edges, h, c, order_parent_receive)
        torch.testing.assert_close(c, expected_c, atol=1e-3, rtol=0)
        torch.testing.assert_close(h, expected_h, atol=1e-3, rtol=0)

class TestNAryTreeLSTM:

    @pytest.fixture
    def tree_lstm(self):
        # Called before each test method
        in_dim = 2
        out_dim = 2
        layer = NAryTreeLSTM(in_dim, out_dim, 2)

        # Define fixed weights as a dict
        weight_dict = {
            'W_iou': [torch.ones(out_dim * 3, in_dim), torch.zeros(out_dim*3)],
            'U_iou': [torch.ones(2, out_dim * 3, in_dim)],
            'W_f': [torch.ones(out_dim, in_dim), torch.zeros(out_dim)],
            'U_f': [torch.ones(2, out_dim, out_dim)],
        }
        layer.set_weights(weight_dict)
        return layer

    @pytest.fixture
    def tree_lstm_2(self):
        # Define new out_dim to better capture dynamics of weights
        in_dim = 2
        out_dim = 3
        layer = NAryTreeLSTM(in_dim, out_dim, 2)

        # Define fixed weights as a dict
        weight_dict = {
            'W_iou': [torch.ones(out_dim * 3, in_dim), torch.zeros(out_dim*3)],
            'U_iou': [torch.ones(2, out_dim, out_dim * 3)],
            'W_f': [torch.ones(out_dim, in_dim), torch.zeros(out_dim)],
            'U_f': [torch.ones(2, out_dim, out_dim)],
        }
        layer.set_weights(weight_dict)
        layer.set_weights({
            'W_f': [torch.tensor([
                [.25, .75],
                [.5, 1],
                [.75, .25]
            ]), torch.zeros(3)],
            'U_f': [torch.tensor([
                [
                    [-1, -.5, -.25],
                    [-1, -.5, -.25],
                    [-.5, -.5, -.5]
                ],
                [
                    [2, 1, -.5],
                    [2, 1, -.5],
                    [-1, -.5, -1]
                ]
            ])]
        })
        return layer

    def test_set_weights_n_ary(self, tree_lstm):
        torch.testing.assert_close(tree_lstm.U_f, torch.ones(2, 2, 2), atol=1e-5, rtol=0)
        torch.testing.assert_close(tree_lstm.W_f.weight, torch.ones(2, 2), atol=1e-5, rtol=0)
        torch.testing.assert_close(tree_lstm.W_iou.weight, torch.ones(6, 2), atol=1e-5, rtol=0)
        torch.testing.assert_close(tree_lstm.U_iou, torch.ones(2, 6, 2), atol=1e-5, rtol=0)

    def test_batched_linear_n_ary(self, tree_lstm):
        """
        Test for single weight matrix
        :param tree_lstm:
        :return:
        """
        # [n_child, in_dim]
        linear_input = torch.tensor([
            [1,1],
            [1,1]
        ], dtype=torch.float)
        # [n_child, in_dim, out_dim]
        test_weights = torch.nn.Parameter(
            torch.tensor([
                [
                    [1,2,3],
                    [1,2,3],
                ],
                [
                    [-1,-2,-3],
                    [-1,-2,-3]
                ],
            ], dtype=torch.float)
        )
        expected_output = torch.tensor([
            [2,4,6],
            [-2,-4,-6]
        ],dtype=torch.float)
        batched_linear_output = NAryTreeLSTM.batched_linear(test_weights, linear_input)
        torch.testing.assert_close(batched_linear_output, expected_output, atol=1e-5, rtol=0)

    def test_stacked_weight_matrix_n_ary(self, tree_lstm):
        """
        Test for stacked weight matrix (e.g. iou computation)
        :param tree_lstm:
        :return:
        """
        linear_input = torch.tensor([
            [1,1],
            [1,1]
        ], dtype=torch.float)
        # [n_child, in_dim, out_dim]
        test_weights = torch.nn.Parameter(
            torch.tensor([
                [
                    [.5,1,1.5,2,2.5,3],
                    [1,2,3,4,5,6],
                ],
                [
                    [1.5,3,4.5,6,7.5,9],
                    [-1,-2,-3,-4,-5,-6]
                ],
            ], dtype=torch.float)
        )
        expected_output = torch.tensor([
            [1.5,3,4.5,6,7.5,9],
            [.5,1,1.5,2,2.5,3]
        ],dtype=torch.float)

        batched_linear_output = NAryTreeLSTM.batched_linear(test_weights, linear_input)
        torch.testing.assert_close(batched_linear_output, expected_output, atol=1e-5, rtol=0)

    def test_batched_linear_n_ary(self, tree_lstm):
        """
        Test for stacked weight matrix (e.g. iou computation)
        :param tree_lstm:
        :return:
        """
        linear_input_h = torch.tensor([
            [1,1],
            [1,1]
        ], dtype=torch.float)
        linear_input_x = torch.tensor([
            [1,1],
            [1,1]
        ], dtype=torch.float)
        # [n_child, out_dim, out_dim]
        test_weights_h = torch.nn.Parameter(
            torch.tensor([
                [
                    [.5,1,1.5,2,2.5,3],
                    [1,2,3,4,5,6],
                ],
                [
                    [1.5,3,4.5,6,7.5,9],
                    [-1,-2,-3,-4,-5,-6]
                ],
            ], dtype=torch.float)
        )
        test_weights_x = torch.nn.Linear(2, 3 * 2)
        with torch.no_grad():
            test_weights_x.weight.copy_(torch.tensor([
                [.5,1,2,3,4,5],
                [.5,1,2,3,4,5],
            ],dtype=torch.float).T)
            test_weights_x.bias.copy_(torch.zeros(2*3))

        expected_output = [
            torch.tensor([[2.5, 5], [1.5, 3]], dtype=torch.float),
            torch.tensor([[8.5, 12], [5.5, 8]], dtype=torch.float),
            torch.tensor([[15.5, 19], [10.5, 13]], dtype=torch.float),
        ]

        batched_linear_output = NAryTreeLSTM.batched_chunked_linear(
            weight_x=test_weights_x, weight_h=test_weights_h,
            input_x=linear_input_x, input_h=linear_input_h,
            chunks=3, dim=1
        )
        for output_tensor, expected_tensor in zip(batched_linear_output, expected_output):
            torch.testing.assert_close(output_tensor, expected_tensor, atol=1e-5, rtol=0)

    def test_get_iou_n_ary(self, tree_lstm_2):
        # linear_input_h = torch.tensor([
        #     [1, 1, 1],
        #     [1, 1, 1]
        # ], dtype=torch.float)
        # linear_input_x = torch.tensor([
        #     [1,1],
        #     [1,1]
        # ], dtype=torch.float)
        # # test_weights_h = torch.nn.Parameter(
        # #     torch.tensor([
        # #         [
        # #             [.5, 1, 1.5, 2, 2.5, 3],
        # #             [1, 2, 3, 4, 5, 6],
        # #             [-2, -1, -.5, 0, 1, 2]
        # #         ],
        # #         [
        # #             [1.5, 3, 4.5, 6, 7.5, 9],
        # #             [-1, -2, -3, -4, -5, -6],
        # #             [1.5, 3, 1.5, 0, 1.5, 3]
        # #         ],
        # #     ], dtype=torch.float)
        # # )
        # # [n_child, out_dim*3, out_dim]
        # # test_weights_h = torch.nn.Parameter(
        # #     torch.tensor([
        # #         [
        # #             [.5, 1, 0], [1, 2, 0], [-2, 1, .5],
        # #             [1.5, 2, -3], [3, 4, -2], [-.5, 0, -.5],
        # #             [2.5, 3, -1], [5, 6, -7], [1, 2, -1]
        # #         ],
        # #         [
        # #             [1.5, 3, -1], [-1, -2, 1], [1.5, 3, -2],
        # #             [4.5, 6, -7], [-3, -4, -2], [1.5, 0, .5],
        # #             [7.5, 9, -8], [-5, -6, 4], [1.5, 3, 0]
        # #         ]
        # #     ])
        # # )
        # # print("TRANSPOSE")
        # # print(torch.transpose(test_weights_h, 1,2))
        # test_weights_h = torch.nn.Parameter(
        #     torch.tensor([
        #         [
        #
        #             [.5, 1, 0, 1.5, 2, -3, 2.5, 3, -1],
        #             [1, 2, 0, 3, 4, -2, 5, 6, -7],
        #             [-2, 1, .5, -.5, 0, -.5, 1, 2, -1]
        #         ],
        #         [
        #             [1.5, 3, -1, -1, -2, 1, 1.5, 3, -2],
        #             [4.5, 6, -7, -3, -4, -2, 1.5, 0, .5],
        #             [7.5, 9, -8, -5, -6, 4, 1.5, 3, 0]
        #         ]
        #     ])
        # )
        #
        #
        # # [n_child, in_dim, out_dim * 3]
        # # test_weights_x = torch.nn.Linear(2, 3 * 2)
        # # print(tree_lstm_2.W_iou)
        # # tree_lstm_2.set_weights({
        # #     'W_iou': [torch.tensor([
        # #         [.25, .75, 0.5, -1, .25, .75],
        # #         [.5, 1, 0.25, -0.5, .5, 1],
        # #         [.75, .25, 0, 1, .75, .25]
        # #     ]), torch.zeros(9)]
        # # })
        # tree_lstm_2.set_weights({
        #     'W_iou': [
        #         # torch.tensor([
        #         # [.25, .75, 0.5, -1, .25, .75],
        #         # [.5, 1, 0.25, -0.5, .5, 1],
        #         # [.75, .25, 0, 1, .75, .25]
        #         # ]),
        #         torch.tensor([
        #             [.25, .75],
        #             [.5, 1],
        #             [.75, .25],
        #             [0.5, -1],
        #             [0.25, -.5],
        #             [0, 1],
        #             [.25, .75],
        #             [.5, 1],
        #             [.75, .25]
        #         ]),
        #         torch.zeros(9)
        #     ],
        #     'U_iou': [test_weights_h]
        # })
        #
        # output = tree_lstm_2.get_iou(x = linear_input_x, h_j = linear_input_h)
        # print(output)
        # assert 5==3

        pass

    def test_message_n_ary(self, tree_lstm_2):
        #   2
        #  / \
        # 0   1  (0 -> 2) (1 -> 2)
        x_j = torch.tensor([[1, 0], [.5, .5]])
        x_i = torch.tensor([[2, 2], [2, 2]], dtype=torch.float)
        # Suppose a hidden state IS defined
        h_j = torch.tensor([[1, 2, 1], [.5, .25, .5]], dtype=torch.float)
        c_j = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
        index = torch.tensor([2, 2])
        # Chunked i_o_u tensor
        expected_i_o_u = torch.tensor([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
        ])
        expected_f_c = torch.tensor([
            [0.3648, 1.4622, 1.3584],
            [0.9526, 0.9707, 0.7549]
        ])

        h_j_updated, f_c, p_iou, x_i = tree_lstm_2.message(x_j, x_i, h_j, c_j, index)

        torch.testing.assert_close(f_c, expected_f_c, atol=1e-4, rtol=0)
        torch.testing.assert_close(p_iou, expected_i_o_u, atol=1e-4, rtol=0)


    def test_aggregate_n_ary(self, tree_lstm_2):
        input_index = torch.tensor([2,2])
        h_j = torch.tensor([[.5, 0, -1], [-1, 1, .5]], dtype=torch.float)
        f_c = torch.tensor([[1, -2, .5], [-1, -2, 1.5]], dtype=torch.float)
        x_i = torch.tensor([[2, 2], [2, 2]], dtype=torch.float)
        p_iou = torch.tensor([
            [.5,-.5, 1, -.5, 1, .5, 1, 1, -.5],
            [1, -.5, 1, .5, 1, .5, -1, -.5, 2]
        ], dtype=torch.float)
        tree_lstm_2.partial_dense_mapping = {2: 0}
        expected_f_c_sum = torch.tensor([[0, -4, 2]], dtype=torch.float)
        expected_p_iou_sum = torch.tensor([[1.5, -1, 2, 0, 2, 1, 0, 0.5, 1.5]],dtype=torch.float)
        # noinspection PyTypeChecker
        output_h_j, f_c_sum, p_iou_sum = tree_lstm_2.aggregate(
            (h_j, f_c, p_iou, x_i), input_index
        )
        torch.testing.assert_close(output_h_j, h_j, atol=1e-4, rtol=0)
        torch.testing.assert_close(f_c_sum, expected_f_c_sum, atol=1e-4, rtol=0)
        torch.testing.assert_close(p_iou_sum, expected_p_iou_sum, atol=1e-4, rtol=0)


    def test_forward_n_ary(self, tree_lstm_2):
        pass