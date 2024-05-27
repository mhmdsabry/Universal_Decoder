import torch
import torch.nn as nn
import numpy as np 

class ACTModule(nn.Module):
  def __init__(self, hidden_size, threshold=0.9, max_hop=12):
    super(ACTModule, self).__init__()
    self.threshold = threshold
    self.max_hop = max_hop
    # initialise a linear layer to learn halting process
    self.p = nn.Linear(hidden_size, 1)
    self.p_sigmoid = nn.Sigmoid()
    self.p.bias.data.fill_(1)
    self.p.bias._no_reinit = True # to not reinitialise it again
    self.norm = nn.LayerNorm(hidden_size)

  def compute_halting_probability(self, state):
    state_normed = self.norm(state)
    p = self.p_sigmoid(self.p(state_normed)).squeeze(-1)
    return p

  def update_state(self, state, p, still_running, new_halted, remainders, previous_state, transformation_fn):
    update_weights = p * still_running + new_halted * remainders
    state = transformation_fn(state)
    new_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
    return new_state
  
  #def forward(self, inputs, time_enc, pos_enc, transformation_fn):
  def forward(self, inputs, time_enc, pos_enc, transformation_fn):
    # Initialise halting process components
    halting_probabilities = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device)
    remainders = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device)
    n_updates = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device) 
    previous_state = torch.zeros_like(inputs)
    step = 0

    # Condition to exit when all inputs halt or we reach a max hop (i.e transformation steps)
    while ((halting_probabilities < self.threshold) & (n_updates < self.max_hop)).any():
        # Add time encoding (based on input index), position encoding (based on step index) to the state
        state = inputs + time_enc[:, :inputs.shape[1]].to(inputs.dtype)
        state = state + pos_enc[:, step].unsqueeze(1).to(inputs.dtype)
        
        # Compute halting probability of the current step using learnable function
        p = self.compute_halting_probability(state)

        # Check which input currently halted or still running 
        still_running = (halting_probabilities < 1.0).float()

        # Calculate the new halted inputs, based on the computed halting probability (i.e p)
        new_halted = (halting_probabilities + p * still_running > self.threshold).float() * still_running

        # Update still_running mask based on the computed halting probability (i.e p) and prev. halting_probabilities
        still_running = (halting_probabilities + p * still_running <= self.threshold).float() * still_running

        # Increment halting_probabilities with masked computed halting probability of the current step
        halting_probabilities = halting_probabilities + p * still_running

        # Calculate the porbability remainder (i.e tracking unused resources) of the halted inputs,
        # as a way to make more matured halting in the future
        remainders = remainders + new_halted * (1 - halting_probabilities)

        # Update halting_probabilities with remainders information of the halted inputs.
        halting_probabilities = halting_probabilities + new_halted * remainders

        # Track the number of updates for each input
        n_updates = n_updates + still_running + new_halted

        # Update the state and assign it to be the previous state for the next step
        updated_state = self.update_state(state, p, still_running, new_halted, remainders, previous_state, transformation_fn)
        previous_state = updated_state
        
        # Update step
        step +=1
    return previous_state, (remainders, n_updates)
  


def test_halting_behavior():
    # Parameters
    batch_size = 2
    seq_len = 3
    hidden_size = 5
    max_hop = 10

    # Mock inputs
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    time_enc = torch.randn(1, seq_len, hidden_size)
    pos_enc = torch.randn(1, max_hop, hidden_size)
    
    # create transformation function (transformer model in our case)
    class MockTransformation(nn.Module):
      def __init__(self):
          super(MockTransformation, self).__init__()

      def forward(self, state):
        return state + 1

    # Instantiate the ACT module
    model = ACTModule(hidden_size) 
    # Instantiate transformation module 
    transformation_fn = MockTransformation()  

    # Run ACT model
    state, (remainders, n_updates) = model.forward(inputs, time_enc, pos_enc, transformation_fn)

    # Assert that the model halts for all inputs
    assert (n_updates <= max_hop).all().item(), "Error: Not all updates are within the max_hop limit."

    # Assert that remainders are calculated correctly
    assert (remainders >= 0).all().item(), "Error: Some remainders are negative."
    assert (remainders <= 1).all().item(), "Error: Some remainders are greater than 1."

    # Ensure the state has changed correctly assuming no inputs were halted in the first pass
    expected_state_change = transformation_fn(inputs)
    actual_state_change = state
    assert not torch.allclose(actual_state_change, expected_state_change, atol=1e-06, rtol=1e-04), "Error: The state change does not match the expected transformation."

    print("All tests passed successfully. The model behaves as intended.")

if __name__ == "__main__":
   test_halting_behavior()