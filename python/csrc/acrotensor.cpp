/* acrotensor.cpp: PyTorch binding for Acrotensor
 * 
 */

#include <vector>
#include <string>
#include <assert.h>

#include <torch/extension.h>

#include "AcroTensor.hpp"

using namespace torch::autograd;

class AcroEinsumFunction : public torch::autograd::Function<AcroEinsumFunction> {
    public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        std::string einstr,
        torch::Tensor op0,
        torch::optional<torch::Tensor> op1,
        torch::optional<torch::Tensor> op2,
        torch::optional<torch::Tensor> op3
    ) {
        std::vector<torch::Tensor> operands;
        operands.reserve(4);  // how many is our max
        // ugly, but hey:
        operands.push_back(op0);
        if (op1.has_value())
            operands.push_back(op1.value());
        if (op2.has_value())
            operands.push_back(op2.value());
        if (op3.has_value())
            operands.push_back(op3.value());
        // end hack

        // parse einstr
        std::vector<std::vector<char>> operand_labels(operands.size() + 1);
        size_t op_index = 0;
        for (auto it = einstr.begin(); it < einstr.end(); it++) {
            switch (*it) {
                case ',':
                case '-':
                    op_index++;
                    break;
                case '>':
                    break;
                default:
                    operand_labels[op_index].push_back(*it);
            }
        }
        // cur_labels now holds the labels of the output
        assert (operands.size() == op_index + 1);

        // Make backwards einstrings
        // std::vector<std::string> backward_einstrs(operands.size());
        // ctx->saved_data["backward_einstrs"] = backward_einstrs;
        ctx->save_for_backward(operands);

        // Make acrotensor einstring
        std::vector<char> acrostr(200); // some number
        acrostr.push_back('Z');
        // last labels are output labels:
        for (auto it = operand_labels.back().begin(); it < operand_labels.back().end(); it++) {
            acrostr.push_back('_');
            acrostr.push_back(*it);
        }
        acrostr.push_back(' ');
        acrostr.push_back('=');
        // operands
        for (op_index = 0; op_index < operands.size(); op_index++) {
            acrostr.push_back(' ');
            acrostr.push_back('A' + op_index);
            for (auto it = operand_labels[op_index].begin(); it < operand_labels[op_index].end(); it++) {
                acrostr.push_back('_');
                acrostr.push_back(*it);
            }
        }

        // Make acro::Tensor s
        std::vector<acro::Tensor> operands_acro(operands.size());
        std::vector<acro::Tensor*> operands_acro_ptr(operands.size());
        op_index = 0;
        std::vector<int> cur_sizes;
        acro::Tensor cur_tensor;
        for (auto it = operands.begin(); it < operands.end(); it++, op_index++) {
            // void Init(std::vector<int> &dims, double *hdata=nullptr, double *ddata=nullptr, bool ongpu=false);
            cur_sizes = std::vector<int>(it->sizes().begin(), it->sizes().end());
            operands_acro[op_index] = acro::Tensor(
                cur_sizes,
                (*it).contiguous().data_ptr<double>()
            );
            operands_acro_ptr[op_index] = &operands_acro[op_index];
        }

        auto out = torch::empty({3, 3});
        acro::Tensor out_acro = acro::Tensor(
            3, 3,
            out.data_ptr<double>()
        );

        // Run contraction
        std::string acrostr_final(acrostr.begin(), acrostr.end());
        AcroEinsumFunction::TE(
            acrostr_final,
            &out_acro, operands_acro_ptr
        );

        return out;
    }

    static tensor_list backward(
        AutogradContext *ctx,
        tensor_list grad_outputs
    ) {
        auto grad_out = grad_outputs[0];
        std::vector<torch::Tensor> grads;
        return {grad_out};
    }

    private:
    static acro::TensorEngine TE;
};

acro::TensorEngine AcroEinsumFunction::TE = acro::TensorEngine("CPUInterpreted");


torch::Tensor einsum(
    std::string einstr,
    torch::Tensor op0,
    torch::optional<torch::Tensor> op1,
    torch::optional<torch::Tensor> op2,
    torch::optional<torch::Tensor> op3
) {
    return AcroEinsumFunction::apply(einstr, op0, op1, op2, op3);
}

static auto registry = torch::RegisterOperators().op("acrotensor::einsum", &einsum);
