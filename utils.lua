function nn.utils.recursiveCudaOn(param, device)
   if torch.type(param) == 'table' or
      torch.isTypeOf(param, 'nn.Module') or
      torch.isTypeOf(param, 'nn.Criterion') then
      for k, v in pairs(param) do
         param[k] = nn.utils.recursiveCudaOn(v, device)
      end
   elseif torch.isTensor(param) then
      param = param:cudaOn(device)
   end
   return param
end
