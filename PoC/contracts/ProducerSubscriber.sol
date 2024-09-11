// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract ProducerConsumerContract is AccessControl {
    
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    struct Producer {
        bool isRegistered;
        bool flag;
    }

    struct Consumer {
        uint256 subscriptionEnd;
    }

    mapping(address => Producer) public producers;
    mapping(address => mapping(address => Consumer)) public subscriptions; // Producer -> Consumer -> Subscription Info

    modifier onlySubscribed(address producer) {
        require(block.timestamp < subscriptions[producer][msg.sender].subscriptionEnd, "Subscription expired");
        _;
    }

    constructor(address admin) {
        _grantRole(ADMIN_ROLE, admin);
        _setRoleAdmin(ADMIN_ROLE, ADMIN_ROLE);
    }

    // Admin registers a producer
    function registerProducer(address producerAddress) external onlyRole(ADMIN_ROLE) {
        producers[producerAddress].isRegistered = true;
    }

    // Producer sets their flag
    function setFlag(bool _flag) external {
        require(producers[msg.sender].isRegistered, "Only registered producers can set flag");
        producers[msg.sender].flag = _flag;
    }

    // Admin subscribes a consumer to a producer
    function subscribeConsumer(address producerAddress, address consumerAddress, uint256 duration) external onlyRole(ADMIN_ROLE) {
        require(producers[producerAddress].isRegistered, "Producer must be registered");
        subscriptions[producerAddress][consumerAddress].subscriptionEnd = block.timestamp + duration;
    }

    // Consumer reads the producer's flag if subscription is active
    function readFlag(address producerAddress) external view onlySubscribed(producerAddress) returns (bool) {
        return producers[producerAddress].flag;
    }

    // Transfer admin role
    function transferAdminRole(address newAdmin) external {
        grantRole(ADMIN_ROLE, newAdmin);
        revokeRole(ADMIN_ROLE, msg.sender);
    }
}
