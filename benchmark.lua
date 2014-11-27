require 'sys'
require 'cunn'

steps = 4 -- nb of steps in loop to average perf
ops = 2 -- ops per point

runs = {
   {
      -- first layer
      ni = 3,
      no = 96,
      kw = 11,
      kh = 11,
      iw = 128,
      ih = 128,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      -- second layer
      ni = 64,
      no = 128,
      kw = 9,
      kh = 9,
      iw = 64,
      ih = 64,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      -- third layer
      ni = 128,
      no = 128,
      kw = 9,
      kh = 9,
      iw = 32,
      ih = 32,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      -- fourth layer
      ni = 128,
      no = 128,
      kw = 7,
      kh = 7,
      iw = 16,
      ih = 16,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {  -- layers with small inputs/kernels, seen at the lower ends of the network
      ni = 384,
      no = 384,
      kw = 3,
      kh = 3,
      iw = 13,
      ih = 13,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      -- first layer, tiny batch
      ni = 3,
      no = 96,
      kw = 11,
      kh = 11,
      iw = 128,
      ih = 128,
      bs = 16,
      dw = 1,
      dh = 1,
   },
   {
      -- second layer, tiny batch
      ni = 64,
      no = 128,
      kw = 9,
      kh = 9,
      iw = 64,
      ih = 64,
      bs = 16,
      dw = 1,
      dh = 1,
   },
}

for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   print('')
   print('CONFIG: input = ' .. ni..'x'..iw..'x'..ih..' * ker = ' .. ni..'x'..no..'x'..kw..'x'..kh .. ' (bs = '..bs..', stride = ' .. dw .. ')')

   n1 = nn.SpatialConvolutionCUDA(ni,no,kw,kh,dw,dh):cuda()
   n2 = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   -- n3 = nn.SpatialConvolutionMM_BHWD(ni,no,kw,kh,dw,dh):cuda()

   i1 = torch.randn(ni, ih, iw, bs):cuda()
   i2 = torch.randn(bs, ni, ih, iw):cuda()
   -- i3 = torch.randn(bs, ih, iw, ni):cuda()

   o1 = n1:forward(i1)
   o2 = n2:forward(i2)
   -- o3 = n3:forward(i3)

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      o1 = n1:updateOutput(i1)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('DHWB:updateOutput(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      o2 = n2:updateOutput(i2)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('BDHW:updateOutput(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')
   
   -- cutorch.synchronize()
   -- sys.tic()
   -- for t = 1,steps do
   --    o3 = n3:updateOutput(i3)
   -- end
   -- cutorch.synchronize()
   -- tm = sys.toc()/steps
   -- print('BHWD:updateOutput(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   collectgarbage()
   
   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      n1:updateGradInput(i1, o1)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('DHWB:updateGradInput(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      n2:updateGradInput(i2, o2)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('BDHW:updateGradInput(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')
   
   collectgarbage()
   
   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      n1:accGradParameters(i1, o1)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('DHWB:accGradParameters(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      n2:accGradParameters(i2, o2)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('BDHW:accGradParameters(): ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')
end

print('')
