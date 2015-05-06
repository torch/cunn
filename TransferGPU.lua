local TransferGPU, parent = torch.class('nn.TransferGPU', 'nn.Module')

function TransferGPU:__init(srcDev, dstDev)
   parent.__init(self)

   self.srcDev = srcDev
   self.dstDev = dstDev

   self.gradInput = torch.CudaTensorOn(srcDev)
   self.output    = torch.CudaTensorOn(dstDev)

end

function TransferGPU:updateOutput(input)
   assert(input:getDevice() == self.srcDev, 
      string.format("input on wrong device, expected %d got %d", input:getDevice(), self.srcDev))
   if self.srcDev ~= self.dstDev then
      self.output:resize(input:size()):copy(input)
   else
      self.output = input
   end
   return self.output
end

function TransferGPU:updateGradInput(input, gradOutput)
   assert(input:getDevice() == self.srcDev, 
      string.format("input on wrong device, expected %d got %d", input:getDevice(), self.srcDev))
   assert(gradOutput:getDevice() == self.dstDev,
      string.format("gradOutput on wrong device, expected %d got %d", gradOutput:getDevice(), self.dstDev))
   if self.srcDev ~= self.dstDev then
      self.gradInput:resize(gradOutput:size()):copy(gradOutput)
   else
      self.gradInput = gradOutput
   end
   return self.gradInput
end
