// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract HealthFederatedToken is ERC20 {
    
    constructor() ERC20("HealthFederatedToken", "HFD") {
        // initial supply hard coded to 1,000,000
        _mint(msg.sender, 1_000_000 * (10 ** 18));
    }

}