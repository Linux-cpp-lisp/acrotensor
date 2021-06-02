/* acrotensor.cpp: PyTorch binding for Acrotensor
 * 
 */

#include <vector>
#include <string>
#include <assert.h>

#include <torch/extension.h>

#include "AcroTensor.hpp"

using namespace torch::autograd;


// TODO: use fixed size arrays?

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
                case '-':
                    it++;  // consume the >
                    TORCH_CHECK(*it == '>', "Malformed formula (- without >)");
                    // continue on, since now we'll have to get the output labels
                case ',':
                    TORCH_CHECK(operand_labels[op_index].size() > 0, "Cannot have empty index labels");
                    op_index++;
                    break;
                default:
                    TORCH_CHECK(std::islower(*it), "Unallowed character in einsum formula (note that ellipsis is not supported)");
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
        // We also make the output size here, since we're looping anyway
        std::vector<char> acrostr;
        std::unordered_map<char, int64_t> dim_sizes;
        acrostr.reserve(200); // some number, TODO
        acrostr.push_back('Z'); // output variable; this assumes less than 26 but we're gueranteed that
        // last labels are output labels:
        for (auto it = operand_labels.back().begin(); it < operand_labels.back().end(); it++) {
            acrostr.push_back('_');
            acrostr.push_back(*it);
        }
        acrostr.push_back(' ');
        acrostr.push_back('=');
        // operands
        size_t label_i;
        for (op_index = 0; op_index < operands.size(); op_index++) {
            acrostr.push_back(' ');
            acrostr.push_back('A' + op_index);
            label_i = 0;
            for (auto it = operand_labels[op_index].begin(); it < operand_labels[op_index].end(); it++) {
                acrostr.push_back('_');
                acrostr.push_back(*it);
                // Save or check dimension size
                if (dim_sizes.find(*it) == dim_sizes.end()) {
                    // add it
                    dim_sizes[*it] = operands[op_index].size(label_i);
                }
                else {
                    // check it
                    TORCH_CHECK(operands[op_index].size(label_i) == dim_sizes[*it], "Inconsistant operand dimensions");
                }
                label_i++;
            }
        }

        // Make acro::Tensor s
        std::vector<acro::Tensor> operands_acro(operands.size());
        std::vector<acro::Tensor*> operands_acro_ptr(operands.size());
        op_index = 0;
        std::vector<int> cur_sizes;
        acro::Tensor cur_tensor;
        for (auto it = operands.begin(); it < operands.end(); it++, op_index++) {
            // void Init(std::vector<int> &dims, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
            cur_sizes = std::vector<int>(it->sizes().begin(), it->sizes().end());
            if ((*it).device() == torch::kCUDA) {
                operands_acro[op_index].Init(
                    cur_sizes,
                    nullptr, // hdata
                    (*it).contiguous().data_ptr<float>(), // ddata
                    true // ongpu
                );
            }
            else {
                operands_acro[op_index].Init(
                    cur_sizes,
                    (*it).contiguous().data_ptr<float>() // hdata
                );
            }
            operands_acro_ptr[op_index] = &operands_acro[op_index];
        }

        // Make output shape
        std::vector<int64_t> outshape;
        std::vector<int> outshape_acro;  // TODO: update to use int64 in acro
        for (auto it = operand_labels.back().begin(); it < operand_labels.back().end(); it++) {
            outshape.push_back(dim_sizes[*it]);
            outshape_acro.push_back(outshape.back());
        }

        auto options = torch::TensorOptions().device(op0.device());
        auto out = torch::empty(outshape, options);
        acro::Tensor out_acro;
        if (op0.device() == torch::kCUDA) {
            out_acro.Init(
                outshape_acro,
                nullptr, // hdata
                out.data_ptr<float>(), // ddata
                true
            );
        }
        else {
            out_acro.Init(
                outshape_acro,
                out.data_ptr<float>() // hdata
            );
        }

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
